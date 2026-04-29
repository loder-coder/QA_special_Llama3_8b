from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from json import JSONDecoder
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from preprocess.clean import clean_text, validate_columns


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
TEXT_SIGNAL = re.compile(r"[0-9A-Za-z가-힣ぁ-んァ-ン一-龥]")
DEFAULT_CONFIG_PATH = "preprocess/scoring_config.yaml"


@dataclass(frozen=True)
class ScoringConfig:
    alpha: float = 0.55
    beta: float = 0.35
    gamma: float = 0.10
    similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llama_model_path: str = "meta-llama/Meta-Llama-3-8B"
    chunk_size: int = 1000
    sim_batch_size: int = 128
    ppl_batch_size: int = 2
    max_length: int = 2048
    gold_ratio: float = 0.10
    num_workers: int = max(1, os.cpu_count() or 1)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required when using preprocess/scoring_config.yaml") from exc
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"scoring config must be a mapping: {path}")
    return payload


def load_scoring_config(config_path: str = DEFAULT_CONFIG_PATH) -> ScoringConfig:
    payload = _load_yaml(Path(config_path))

    def pick(name: str, default: object, cast: type) -> object:
        env_name = f"ALTONG_SCORING_{name.upper()}"
        value = os.getenv(env_name, payload.get(name, default))
        return cast(value)

    num_workers = pick("num_workers", ScoringConfig.num_workers, int)
    if num_workers <= 0:
        num_workers = max(1, os.cpu_count() or 1)

    return ScoringConfig(
        alpha=pick("alpha", ScoringConfig.alpha, float),
        beta=pick("beta", ScoringConfig.beta, float),
        gamma=pick("gamma", ScoringConfig.gamma, float),
        similarity_model=str(os.getenv("ALTONG_SCORING_SIMILARITY_MODEL", payload.get("similarity_model", ScoringConfig.similarity_model))),
        llama_model_path=str(os.getenv("ALTONG_SCORING_LLAMA_MODEL_PATH", payload.get("llama_model_path", ScoringConfig.llama_model_path))),
        chunk_size=pick("chunk_size", ScoringConfig.chunk_size, int),
        sim_batch_size=pick("sim_batch_size", ScoringConfig.sim_batch_size, int),
        ppl_batch_size=pick("ppl_batch_size", ScoringConfig.ppl_batch_size, int),
        max_length=pick("max_length", ScoringConfig.max_length, int),
        gold_ratio=pick("gold_ratio", ScoringConfig.gold_ratio, float),
        num_workers=num_workers,
    )


def _iter_json_with_decoder(path: Path) -> Iterator[dict]:
    decoder = JSONDecoder()
    buffer = ""
    started = False
    with path.open("r", encoding="utf-8") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk and not buffer.strip():
                break
            buffer += chunk
            while True:
                buffer = buffer.lstrip()
                if not started:
                    if buffer.startswith("["):
                        buffer = buffer[1:]
                        started = True
                    elif buffer:
                        raise ValueError("raw data must be a JSON array or JSONL file")
                    else:
                        break
                buffer = buffer.lstrip()
                if buffer.startswith("]"):
                    return
                if buffer.startswith(","):
                    buffer = buffer[1:].lstrip()
                try:
                    item, index = decoder.raw_decode(buffer)
                except ValueError:
                    if not chunk:
                        raise
                    break
                if not isinstance(item, dict):
                    raise ValueError("each raw data item must be a JSON object")
                yield item
                buffer = buffer[index:]
            if not chunk:
                break


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("each JSONL line must be a JSON object")
            yield payload


def load_data(path: str = "data/raw.json") -> Iterator[dict]:
    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(f"raw data file not found: {path}")
    if raw_path.suffix.lower() == ".jsonl":
        yield from _iter_jsonl(raw_path)
        return
    try:
        import ijson
    except ImportError:
        yield from _iter_json_with_decoder(raw_path)
        return
    with raw_path.open("rb") as file:
        for item in ijson.items(file, "item"):
            if not isinstance(item, dict):
                raise ValueError("each raw data item must be a JSON object")
            yield item


def load_raw_data(path: str = "data/raw.json") -> Iterator[dict]:
    return load_data(path)


def _has_invalid_training_text(value: object) -> bool:
    text = clean_text(value)
    if not text:
        return True
    if CONTROL_CHARS.search(text):
        return True
    if not TEXT_SIGNAL.search(text):
        return True
    return False


def _clean_chunk(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    validate_columns(df)
    df = df.dropna(subset=["question", "answer"])
    df = df[~df["question"].map(_has_invalid_training_text)]
    df = df[~df["answer"].map(_has_invalid_training_text)]
    if df.empty:
        return []
    cleaned = df.copy()
    cleaned["question"] = cleaned["question"].map(clean_text)
    cleaned["answer"] = cleaned["answer"].map(clean_text)
    cleaned["category"] = cleaned["category"].map(clean_text)
    cleaned["views"] = pd.to_numeric(cleaned["views"], errors="coerce").fillna(0)
    cleaned["donation"] = pd.to_numeric(cleaned["donation"], errors="coerce").fillna(0)
    cleaned = cleaned[~cleaned["question"].map(_has_invalid_training_text)]
    cleaned = cleaned[~cleaned["answer"].map(_has_invalid_training_text)]
    cleaned = cleaned.reset_index(drop=True)
    return cleaned.to_dict("records")


def _chunked(iterator: Iterable[dict], chunk_size: int) -> Iterator[list[dict]]:
    chunk = []
    for item in iterator:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _iter_clean_rows(raw_path: str, config: ScoringConfig, skip_rows: int = 0) -> Iterator[dict]:
    chunks = _chunked(load_data(raw_path), config.chunk_size)
    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        for cleaned_chunk in executor.map(_clean_chunk, chunks):
            for row in cleaned_chunk:
                if skip_rows > 0:
                    skip_rows -= 1
                    continue
                yield row


class PplScorer:
    def __init__(self, model_path: str, max_length: int) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    def score_batch(self, questions: list[str], answers: list[str]) -> list[float]:
        texts = [f"Question: {question}\nAnswer: {answer}" for question, answer in zip(questions, answers)]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.model.device)
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
            logits = outputs.logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            if shifted_labels.size(1) == 0:
                return [0.0 for _ in texts]
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                shifted_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(shifted_labels.size())
            token_counts = (shifted_labels != -100).sum(dim=1).clamp_min(1)
            sample_loss = loss.sum(dim=1) / token_counts
            ppl = torch.exp(sample_loss).detach().float().cpu().numpy()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return [float(1.0 / max(value, 1e-6)) for value in ppl]


def _length_penalty(answer: str) -> float:
    length = len(answer)
    if length < 20:
        return -1.0
    if length > 1000:
        return max(-1.0, 1.0 - ((length - 1000) / 1000.0))
    return 1.0


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {"processed_rows": 0, "scores": []}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_checkpoint(path: Path, processed_rows: int, scores: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump({"processed_rows": processed_rows, "scores": scores}, file, ensure_ascii=False, indent=2)


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_report(path: Path, gold_scores: list[float], bronze_scores: list[float]) -> None:
    def stats(values: list[float]) -> str:
        if not values:
            return "count=0, avg=0.000000, min=0.000000, max=0.000000"
        return (
            f"count={len(values)}, "
            f"avg={float(np.mean(values)):.6f}, "
            f"min={float(np.min(values)):.6f}, "
            f"max={float(np.max(values)):.6f}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("Q_score preprocessing report\n")
        file.write(f"Gold: {stats(gold_scores)}\n")
        file.write(f"Bronze: {stats(bronze_scores)}\n")


def _score_rows(rows: list[dict], sim_model: SentenceTransformer, ppl_scorer: PplScorer, config: ScoringConfig) -> list[dict]:
    questions = [str(row["question"]) for row in rows]
    answers = [str(row["answer"]) for row in rows]
    q_embeddings = sim_model.encode(
        questions,
        batch_size=config.sim_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    a_embeddings = sim_model.encode(
        answers,
        batch_size=config.sim_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    similarities = np.sum(q_embeddings * a_embeddings, axis=1)
    inverse_ppls: list[float] = []
    for start in range(0, len(rows), config.ppl_batch_size):
        end = start + config.ppl_batch_size
        inverse_ppls.extend(ppl_scorer.score_batch(questions[start:end], answers[start:end]))

    scored_rows = []
    for row, similarity, inverse_ppl in zip(rows, similarities, inverse_ppls):
        penalty = _length_penalty(str(row["answer"]))
        q_score = (config.alpha * float(similarity)) + (config.beta * inverse_ppl) + (config.gamma * penalty)
        enriched = dict(row)
        enriched["sim_qa"] = float(similarity)
        enriched["inverse_ppl"] = inverse_ppl
        enriched["length_penalty"] = penalty
        enriched["score"] = float(q_score)
        scored_rows.append(enriched)
    return scored_rows


def _partition_outputs(scored_path: Path, gold_path: Path, bronze_path: Path, report_path: Path, scores: list[float], gold_ratio: float) -> None:
    if not scores:
        _write_jsonl(gold_path, [])
        _write_jsonl(bronze_path, [])
        _write_report(report_path, [], [])
        return
    gold_count = max(1, math.ceil(len(scores) * gold_ratio))
    threshold = sorted(scores, reverse=True)[gold_count - 1]
    gold_scores: list[float] = []
    bronze_scores: list[float] = []

    gold_line_numbers: set[int] = set()

    def gold_rows() -> Iterator[dict]:
        remaining = gold_count
        with scored_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file):
                row = json.loads(line)
                score = float(row["score"])
                if score >= threshold and remaining > 0:
                    gold_scores.append(score)
                    remaining -= 1
                    gold_line_numbers.add(line_number)
                    yield row

    _write_jsonl(gold_path, gold_rows())

    def bronze_rows() -> Iterator[dict]:
        with scored_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file):
                if line_number in gold_line_numbers:
                    continue
                row = json.loads(line)
                score = float(row["score"])
                bronze_scores.append(score)
                yield row

    _write_jsonl(bronze_path, bronze_rows())
    _write_report(report_path, gold_scores, bronze_scores)


def preprocess(
    raw_path: str = "data/raw.json",
    gold_path: str = "data/Gold.jsonl",
    bronze_path: str = "data/Bronze.jsonl",
    report_path: str = "data/report.txt",
    checkpoint_path: str = "data/preprocess_checkpoint.json",
    scored_path: str = "data/scored.jsonl",
    config_path: str = DEFAULT_CONFIG_PATH,
) -> tuple[Path, Path]:
    config = load_scoring_config(config_path)
    checkpoint_file = Path(checkpoint_path)
    scored_file = Path(scored_path)
    checkpoint = _load_checkpoint(checkpoint_file)
    processed_rows = int(checkpoint.get("processed_rows", 0))
    scores = [float(score) for score in checkpoint.get("scores", [])]
    if processed_rows > 0 and not scored_file.exists():
        processed_rows = 0
        scores = []
        _save_checkpoint(checkpoint_file, processed_rows, scores)
    if processed_rows == 0 and scored_file.exists():
        scored_file.unlink()

    sim_model = SentenceTransformer(config.similarity_model)
    ppl_scorer = PplScorer(config.llama_model_path, config.max_length)
    batch: list[dict] = []

    for row in _iter_clean_rows(raw_path, config, skip_rows=processed_rows):
        batch.append(row)
        if len(batch) < config.chunk_size:
            continue
        scored_rows = _score_rows(batch, sim_model, ppl_scorer, config)
        _append_jsonl(scored_file, scored_rows)
        scores.extend(float(item["score"]) for item in scored_rows)
        processed_rows += len(batch)
        _save_checkpoint(checkpoint_file, processed_rows, scores)
        batch = []

    if batch:
        scored_rows = _score_rows(batch, sim_model, ppl_scorer, config)
        _append_jsonl(scored_file, scored_rows)
        scores.extend(float(item["score"]) for item in scored_rows)
        processed_rows += len(batch)
        _save_checkpoint(checkpoint_file, processed_rows, scores)

    _partition_outputs(scored_file, Path(gold_path), Path(bronze_path), Path(report_path), scores, config.gold_ratio)
    return Path(gold_path), Path(bronze_path)


def run_preprocess(
    raw_path: str = "data/raw.json",
    gold_path: str = "data/Gold.jsonl",
    bronze_path: str = "data/Bronze.jsonl",
    dedup_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[Path, Path]:
    os.environ.setdefault("ALTONG_SCORING_SIMILARITY_MODEL", dedup_model_name)
    return preprocess(raw_path=raw_path, gold_path=gold_path, bronze_path=bronze_path)


if __name__ == "__main__":
    preprocess()
