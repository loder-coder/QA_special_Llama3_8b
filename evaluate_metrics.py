from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _read_json_records(path: str | Path) -> list[dict[str, Any]]:
    data_path = Path(path)
    if not data_path.exists() or data_path.stat().st_size == 0:
        return []

    if data_path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with data_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows

    with data_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        records = payload.get("records") or payload.get("data") or payload.get("items")
        if isinstance(records, list):
            return [item for item in records if isinstance(item, dict)]
        return [payload]
    return []


def _read_existing_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data_path = Path(path)
        if data_path.exists():
            rows.extend(_read_json_records(data_path))
    return rows


def _percent(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return round((float(numerator) / float(denominator)) * 100.0, 6)


def purification_efficiency(raw_rows: list[dict[str, Any]], valid_rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_count = len(raw_rows)
    valid_count = len(valid_rows)
    return {
        "raw_count": raw_count,
        "valid_count": valid_count,
        "efficiency_percent": _percent(valid_count, raw_count),
    }


def category_balance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("category", "")).strip() or "unknown" for row in rows)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        probability = count / total if total else 0.0
        if probability > 0:
            entropy -= probability * math.log(probability)

    category_count = len(counts)
    normalized_entropy = entropy / math.log(category_count) if category_count > 1 else None
    return {
        "record_count": total,
        "category_count": category_count,
        "entropy": round(entropy, 6) if total else None,
        "normalized_entropy": round(normalized_entropy, 6) if normalized_entropy is not None else None,
        "category_counts": dict(sorted(counts.items())),
    }


def _pick_number(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in row:
            continue
        try:
            return float(row[key])
        except (TypeError, ValueError):
            continue
    return None


def _cosine_scores_with_model(rows: list[dict[str, Any]], model_name: str) -> list[tuple[float, float]]:
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("sentence-transformers and numpy are required when --intent-model is used") from exc

    model = SentenceTransformer(model_name)
    texts: list[str] = []
    for row in rows:
        texts.extend([str(row["anchor"]), str(row["positive"]), str(row["negative"])])

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    scores: list[tuple[float, float]] = []
    for index in range(0, len(embeddings), 3):
        anchor = embeddings[index]
        positive = embeddings[index + 1]
        negative = embeddings[index + 2]
        scores.append((float(np.dot(anchor, positive)), float(np.dot(anchor, negative))))
    return scores


def intent_discrimination(rows: list[dict[str, Any]], model_name: str | None = None) -> dict[str, Any]:
    scores: list[float] = []
    used_model = False

    if model_name and rows and all({"anchor", "positive", "negative"} <= set(row) for row in rows):
        for sim_positive, sim_negative in _cosine_scores_with_model(rows, model_name):
            scores.append(max(0.0, sim_positive - sim_negative))
        used_model = True
    else:
        for row in rows:
            sim_positive = _pick_number(row, ("sim_anchor_positive", "sim_ap", "positive_similarity", "sim_positive"))
            sim_negative = _pick_number(row, ("sim_anchor_negative", "sim_an", "negative_similarity", "sim_negative"))
            if sim_positive is None or sim_negative is None:
                continue
            scores.append(max(0.0, sim_positive - sim_negative))

    return {
        "sample_count": len(scores),
        "score": round(sum(scores) / len(scores), 6) if scores else None,
        "used_embedding_model": used_model,
    }


def _result_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    ids: list[str] = []
    for item in value:
        if isinstance(item, dict):
            identifier = item.get("id") or item.get("doc_id") or item.get("document_id") or item.get("metadata_id")
            if identifier is not None:
                ids.append(str(identifier))
        elif item is not None:
            ids.append(str(item))
    return ids


def hit_rate_at_k(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    total = 0
    hits = 0
    for row in rows:
        expected = row.get("expected_id") or row.get("relevant_id") or row.get("answer_id") or row.get("document_id")
        results = row.get("top_k_results") or row.get("results") or row.get("retrieved_ids") or row.get("top_k")
        result_ids = _result_ids(results)
        if expected is None or not result_ids:
            continue
        total += 1
        if str(expected) in result_ids[:k]:
            hits += 1

    return {
        "k": k,
        "query_count": total,
        "hit_count": hits,
        "hit_rate": round(hits / total, 6) if total else None,
    }


def service_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    sources = Counter(str(row.get("source", "unknown")) for row in rows)
    exact_hits = sources.get("redis", 0)
    semantic_hits = sources.get("redis_intent", 0)
    input_blocks = sources.get("blocked", 0)
    output_blocks = sources.get("output_blocked", 0)

    cost_values: list[float] = []
    for row in rows:
        latency = _pick_number(row, ("latency_seconds", "latency", "response_seconds", "elapsed_seconds"))
        if latency is None:
            latency_ms = _pick_number(row, ("latency_ms", "response_ms", "elapsed_ms"))
            if latency_ms is not None:
                latency = latency_ms / 1000.0
        if latency is None:
            continue

        similarity = _pick_number(row, ("similarity", "rag_similarity", "intent_similarity"))
        if similarity is None:
            similarity = max(
                _pick_number(row, ("rag_similarity",)) or 0.0,
                _pick_number(row, ("intent_similarity",)) or 0.0,
            )
        similarity = max(0.0, min(1.0, similarity))
        cost_values.append(latency * (1.0 - similarity))

    return {
        "total_requests": total,
        "source_counts": dict(sorted(sources.items())),
        "cache": {
            "exact_hits": exact_hits,
            "semantic_hits": semantic_hits,
            "total_hit_rate_percent": _percent(exact_hits + semantic_hits, total),
        },
        "security": {
            "input_blocks": input_blocks,
            "output_blocks": output_blocks,
            "security_robustness_percent": _percent(input_blocks + output_blocks, total),
        },
        "response_efficiency": {
            "sample_count": len(cost_values),
            "cost_avg": round(sum(cost_values) / len(cost_values), 6) if cost_values else None,
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    raw_rows = _read_json_records(args.raw)
    valid_rows = _read_existing_records([args.valid])
    if not valid_rows:
        valid_rows = _read_existing_records([args.gold, args.bronze])
    category_rows = valid_rows or _read_existing_records([args.gold])

    report = {
        "data_quality": {
            "purification_efficiency": purification_efficiency(raw_rows, valid_rows),
            "category_balance": category_balance(category_rows),
        },
        "training_metrics": {
            "intent_discrimination": intent_discrimination(_read_json_records(args.triplets), args.intent_model),
            "hit_rate_at_k": hit_rate_at_k(_read_json_records(args.rag_eval), args.k),
        },
        "service_ops": service_metrics(_read_json_records(args.qa_log)),
        "warnings": [],
    }

    if not valid_rows:
        report["warnings"].append("valid data was empty or missing; pass --valid or provide --gold and --bronze")
    if report["training_metrics"]["intent_discrimination"]["sample_count"] == 0:
        report["warnings"].append("intent discrimination needs similarity columns or --intent-model with anchor/positive/negative triplets")
    if report["training_metrics"]["hit_rate_at_k"]["query_count"] == 0:
        report["warnings"].append("hit rate needs a RAG evaluation file with expected_id/relevant_id and top-k results")
    if report["service_ops"]["response_efficiency"]["sample_count"] == 0:
        report["warnings"].append("response efficiency needs latency fields in logs/qa.jsonl")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate altong_ai data, training, and service quality metrics.")
    parser.add_argument("--raw", default="data/raw.json", help="Raw input data JSON or JSONL path.")
    parser.add_argument("--valid", default="data/scored.jsonl", help="Validated preprocessed data path.")
    parser.add_argument("--gold", default="data/Gold.jsonl", help="Gold data path used as fallback valid data.")
    parser.add_argument("--bronze", default="data/Bronze.jsonl", help="Bronze data path used as fallback valid data.")
    parser.add_argument("--triplets", default="data/triplets.json", help="Triplet evaluation data path.")
    parser.add_argument("--intent-model", default=None, help="Optional SentenceTransformer model for triplet scoring.")
    parser.add_argument("--rag-eval", default="data/rag_eval.jsonl", help="RAG evaluation JSON or JSONL path.")
    parser.add_argument("--k", type=int, default=5, help="K value for Hit Rate @K.")
    parser.add_argument("--qa-log", default="logs/qa.jsonl", help="Service log JSONL path.")
    parser.add_argument("--output", default="data/evaluation_metrics.json", help="Report output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
