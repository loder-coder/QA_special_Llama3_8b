from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from preprocess.io import read_records
from security.prompt_guard import is_prompt_injection
from security.sensitive_filter import contains_sensitive_data


def collect_failed_queries(log_path: str = "logs/qa.jsonl") -> pd.DataFrame:
    path = Path(log_path)
    if not path.exists():
        return pd.DataFrame(columns=["query", "answer", "source", "success", "language", "sensitive_data_detected"])

    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("success") is False:
                rows.append(
                    {
                        "query": payload.get("query", ""),
                        "answer": payload.get("answer", ""),
                        "source": payload.get("source", ""),
                        "success": payload.get("success", False),
                        "language": payload.get("language", "ko"),
                        "sensitive_data_detected": payload.get("sensitive_data_detected", False),
                    }
                )
    return pd.DataFrame(rows)


def build_review_candidates(
    log_path: str = "logs/qa.jsonl",
    output_path: str = "data/retrain_candidates.json",
    min_query_length: int = 3,
) -> pd.DataFrame:
    failed = collect_failed_queries(log_path)
    if failed.empty:
        candidates = pd.DataFrame(columns=["query", "language", "count", "latest_answer", "source", "needs_manual_answer", "approved"])
        candidates.to_json(output_path, orient="records", force_ascii=False, indent=2)
        return candidates

    filtered = failed.copy()
    filtered["query"] = filtered["query"].astype(str).str.strip()
    filtered["answer"] = filtered["answer"].astype(str).str.strip()
    filtered["language"] = filtered["language"].astype(str).str.strip()
    filtered = filtered[filtered["query"].str.len() >= min_query_length]
    filtered = filtered[~filtered["query"].map(is_prompt_injection)]
    filtered = filtered[~filtered["query"].map(contains_sensitive_data)]
    filtered = filtered[~filtered["answer"].map(contains_sensitive_data)]
    if "sensitive_data_detected" in filtered.columns:
        filtered = filtered[filtered["sensitive_data_detected"] != True]
    if filtered.empty:
        candidates = pd.DataFrame(columns=["query", "language", "count", "latest_answer", "source", "needs_manual_answer", "approved"])
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        candidates.to_json(output_path, orient="records", force_ascii=False, indent=2)
        return candidates

    counts = Counter((row["query"], row["language"]) for row in filtered.to_dict("records"))
    latest_answer: dict[tuple[str, str], str] = {}
    latest_source: dict[tuple[str, str], str] = {}
    for row in filtered.to_dict("records"):
        key = (row["query"], row["language"])
        latest_answer[key] = row["answer"]
        latest_source[key] = row["source"]

    rows = []
    for (query, language), count in counts.items():
        answer = latest_answer.get((query, language), "")
        rows.append(
            {
                "query": query,
                "language": language,
                "count": count,
                "latest_answer": answer,
                "source": latest_source.get((query, language), ""),
                "needs_manual_answer": not bool(answer),
                "approved": False,
            }
        )

    candidates = pd.DataFrame(rows).sort_values(["count", "query"], ascending=[False, True]).reset_index(drop=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    candidates.to_json(output_path, orient="records", force_ascii=False, indent=2)
    return candidates


def add_approved_queries_to_gold(
    approved_path: str = "data/retrain_approved.json",
    gold_path: str = "data/Gold.jsonl",
    output_path: str = "data/Gold.jsonl",
) -> pd.DataFrame:
    approved_file = Path(approved_path)
    gold_file = Path(gold_path)
    gold = read_records(gold_file) if gold_file.exists() else pd.DataFrame()

    if not approved_file.exists():
        return gold

    approved = read_records(approved_file)
    if approved.empty:
        return gold

    if "approved" not in approved.columns:
        return gold

    approved = approved[approved["approved"] == True]
    if approved.empty:
        return gold

    start_index = len(gold)
    additions = []
    for offset, row in enumerate(approved.to_dict("records"), start=1):
        answer = str(row.get("answer") or row.get("latest_answer") or "").strip()
        query = str(row.get("query", "")).strip()
        if not query or is_prompt_injection(query):
            continue
        if contains_sensitive_data(query) or contains_sensitive_data(answer):
            continue
        if not answer:
            continue
        additions.append(
            {
                "id": f"approved_{start_index + offset:06d}",
                "question": query,
                "answer": answer,
                "category": str(row.get("category", "approved_query")),
                "views": 0,
                "donation": 0,
                "score": 0.0,
            }
        )

    if additions:
        gold = pd.concat([gold, pd.DataFrame(additions)], ignore_index=True)
        gold = gold.drop_duplicates(subset=["question"]).reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(output_path).suffix.lower() == ".jsonl":
        gold.to_json(output_path, orient="records", force_ascii=False, lines=True)
    else:
        gold.to_json(output_path, orient="records", force_ascii=False, indent=2)
    return gold


if __name__ == "__main__":
    build_review_candidates()
