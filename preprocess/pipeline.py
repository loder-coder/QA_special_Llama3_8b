from __future__ import annotations

from pathlib import Path

import pandas as pd

from preprocess.clean import REQUIRED_COLUMNS, clean_dataframe
from preprocess.dedup import remove_duplicates_sbert


def load_raw_data(path: str = "data/raw.json") -> pd.DataFrame:
    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(f"raw data file not found: {path}")
    df = pd.read_json(raw_path)
    if df.empty and not set(REQUIRED_COLUMNS).issubset(df.columns):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    return df


def add_score(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    scored["score"] = scored["views"] * 0.3 + scored["donation"] * 0.7
    return scored


def split_gold_bronze(df: pd.DataFrame, gold_ratio: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()
    ranked = df.sort_values("score", ascending=False).reset_index(drop=True)
    gold_count = max(1, int(len(ranked) * gold_ratio))
    gold = ranked.iloc[:gold_count].reset_index(drop=True)
    bronze = ranked.iloc[gold_count:].reset_index(drop=True)
    return gold, bronze


def run_preprocess(
    raw_path: str = "data/raw.json",
    gold_path: str = "data/gold.json",
    bronze_path: str = "data/bronze.json",
    dedup_model_name: str = "BAAI/bge-small-en-v1.5",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_raw_data(raw_path)
    cleaned = clean_dataframe(df)
    scored = add_score(cleaned)
    deduped = remove_duplicates_sbert(scored, model_name=dedup_model_name, threshold=0.9)
    gold, bronze = split_gold_bronze(deduped)

    Path(gold_path).parent.mkdir(parents=True, exist_ok=True)
    gold.to_json(gold_path, orient="records", force_ascii=False, indent=2)
    bronze.to_json(bronze_path, orient="records", force_ascii=False, indent=2)
    return gold, bronze


if __name__ == "__main__":
    run_preprocess()
