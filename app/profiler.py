"""ProfilerAgent: classify every column and compute per-column statistics.

Semantic types: numeric | categorical | datetime | boolean | text | id
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def _semantic_type(s: pd.Series) -> str:
    n = len(s)
    non_null = s.dropna()
    nunique = non_null.nunique()

    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"

    if pd.api.types.is_numeric_dtype(s):
        # numeric-looking IDs: near-unique integers
        if nunique > 0.95 * max(len(non_null), 1) and pd.api.types.is_integer_dtype(s) and n > 20:
            return "id"
        if 0 < nunique <= 2 and set(non_null.unique()).issubset({0, 1}):
            return "boolean"  # 0< guard: an all-null column is not boolean
        if nunique <= min(10, max(3, n // 50)) and pd.api.types.is_integer_dtype(s):
            return "categorical"  # e.g. rating 1-5, quarter 1-4
        return "numeric"

    # object / string columns
    as_str = non_null.astype(str)
    if len(as_str) == 0:
        return "categorical"
    avg_len = float(as_str.str.len().mean())
    uniq_ratio = nunique / max(len(non_null), 1)
    # long strings are free text — reviews often repeat, so accept either high
    # uniqueness or a reasonable number of distinct long values
    if avg_len > 40 and (uniq_ratio > 0.5 or nunique >= 5):
        return "text"
    if uniq_ratio > 0.95 and n > 20:
        return "id"
    if set(as_str.str.lower().unique()).issubset({"true", "false", "yes", "no", "y", "n", "0", "1"}):
        return "boolean"
    return "categorical"


def _col_profile(s: pd.Series, sem: str) -> dict:
    non_null = s.dropna()
    prof: dict = {
        "semantic_type": sem,
        "dtype": str(s.dtype),
        "count": int(len(s)),
        "nulls": int(s.isna().sum()),
        "null_pct": round(float(s.isna().mean()) * 100, 2),
        "unique": int(non_null.nunique()),
    }
    if sem == "numeric":
        prof.update({
            "min": _num(non_null.min()), "max": _num(non_null.max()),
            "mean": _num(non_null.mean()), "median": _num(non_null.median()),
            "std": _num(non_null.std()),
        })
    elif sem in ("categorical", "boolean"):
        vc = non_null.astype(str).value_counts().head(10)
        prof["top_values"] = [{"value": k, "count": int(v)} for k, v in vc.items()]
    elif sem == "datetime":
        prof.update({"min": str(non_null.min()), "max": str(non_null.max())})
    elif sem == "text":
        lens = non_null.astype(str).str.len()
        prof.update({"avg_length": round(float(lens.mean()), 1),
                     "samples": [str(v)[:200] for v in non_null.head(3)]})
    elif sem == "id":
        prof["samples"] = [str(v) for v in non_null.head(3)]
    return prof


def _num(v):
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


def profile_table(df: pd.DataFrame, table: str) -> dict:
    cols = {}
    for c in df.columns:
        sem = _semantic_type(df[c])
        cols[c] = _col_profile(df[c], sem)
    # Report the true table size (matches meta.tables[].rows); flag when stats came
    # from a sample so consumers don't present sample counts as full-table totals.
    total = int(df.attrs.get("total_rows", len(df)))
    prof = {"table": table, "rows": total, "columns": cols}
    if total > len(df):
        prof["sampled_rows"] = int(len(df))
    return prof


def columns_of_type(profile: dict, sem: str) -> list[str]:
    return [c for c, p in profile["columns"].items() if p["semantic_type"] == sem]
