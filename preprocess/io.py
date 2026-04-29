from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_records(path: str | Path) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        return pd.DataFrame()
    if data_path.stat().st_size == 0:
        return pd.DataFrame()
    if data_path.suffix.lower() == ".jsonl":
        return pd.read_json(data_path, lines=True)
    return pd.read_json(data_path)
