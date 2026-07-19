"""StatsAgent: type-aware pairwise relationship discovery.

numeric <-> numeric   : Pearson + Spearman
categorical <-> numeric: correlation ratio (eta squared)
categorical <-> categorical: Cramér's V
datetime -> numeric   : monthly trend (linear regression slope)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config
from .profiler import columns_of_type


def _eta_squared(df: pd.DataFrame, cat: str, num: str) -> float | None:
    d = df[[cat, num]].dropna()
    if len(d) < 20 or d[cat].nunique() < 2 or d[cat].nunique() > config.MAX_CATEGORICAL_CARDINALITY:
        return None
    groups = d.groupby(cat, observed=True)[num]
    grand_mean = d[num].mean()
    ss_between = float(sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in groups))
    ss_total = float(((d[num] - grand_mean) ** 2).sum())
    if ss_total == 0:
        return None
    return ss_between / ss_total


def _cramers_v(df: pd.DataFrame, a: str, b: str) -> float | None:
    d = df[[a, b]].dropna()
    if len(d) < 20:
        return None
    if d[a].nunique() > config.MAX_CATEGORICAL_CARDINALITY or d[b].nunique() > config.MAX_CATEGORICAL_CARDINALITY:
        return None
    ct = pd.crosstab(d[a], d[b])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return None
    chi2 = sps.chi2_contingency(ct, correction=False)[0]
    n = ct.to_numpy().sum()
    r, k = ct.shape
    denom = n * (min(r, k) - 1)
    if denom == 0:
        return None
    return float(np.sqrt(chi2 / denom))


def analyze_table(df: pd.DataFrame, profile: dict) -> list[dict]:
    findings: list[dict] = []
    table = profile["table"]
    numeric = columns_of_type(profile, "numeric")[: config.MAX_NUMERIC_COLS_FOR_PAIRS]
    categorical = columns_of_type(profile, "categorical")
    datetimes = columns_of_type(profile, "datetime")

    # numeric <-> numeric
    for i in range(len(numeric)):
        for j in range(i + 1, len(numeric)):
            a, b = numeric[i], numeric[j]
            d = df[[a, b]].dropna()
            if len(d) < 20 or d[a].nunique() < 3 or d[b].nunique() < 3:
                continue
            try:
                pearson = float(d[a].corr(d[b], method="pearson"))
                spearman = float(d[a].corr(d[b], method="spearman"))
            except Exception:
                continue
            if np.isnan(pearson) or np.isnan(spearman):
                continue
            strength = max(abs(pearson), abs(spearman))
            if strength >= 0.3:
                findings.append({
                    "kind": "numeric_numeric", "table": table, "columns": [a, b],
                    "pearson": round(pearson, 3), "spearman": round(spearman, 3),
                    "strength": round(strength, 3),
                    "nonlinear": bool(abs(spearman) - abs(pearson) > 0.15),
                })

    # categorical <-> numeric
    for cat in categorical:
        for num in numeric:
            eta = _eta_squared(df, cat, num)
            if eta is not None and eta >= 0.1:
                d = df[[cat, num]].dropna()
                means = d.groupby(cat, observed=True)[num].mean().sort_values(ascending=False)
                findings.append({
                    "kind": "categorical_numeric", "table": table, "columns": [cat, num],
                    "eta_squared": round(eta, 3), "strength": round(eta, 3),
                    "highest_group": {"value": str(means.index[0]), "mean": round(float(means.iloc[0]), 3)},
                    "lowest_group": {"value": str(means.index[-1]), "mean": round(float(means.iloc[-1]), 3)},
                })

    # categorical <-> categorical
    for i in range(len(categorical)):
        for j in range(i + 1, len(categorical)):
            v = _cramers_v(df, categorical[i], categorical[j])
            if v is not None and v >= 0.25:
                findings.append({
                    "kind": "categorical_categorical", "table": table,
                    "columns": [categorical[i], categorical[j]],
                    "cramers_v": round(v, 3), "strength": round(v, 3),
                })

    # datetime trends
    for dt in datetimes[:1]:  # primary time column only
        for num in numeric[:6]:
            d = df[[dt, num]].dropna()
            if len(d) < 30:
                continue
            monthly = d.set_index(dt)[num].sort_index().resample("MS").mean().dropna()
            if len(monthly) < 4:
                continue
            x = np.arange(len(monthly), dtype=float)
            try:
                lr = sps.linregress(x, monthly.to_numpy(dtype=float))
            except Exception:
                continue
            rel_slope = lr.slope / (abs(monthly.mean()) + 1e-9)
            if abs(lr.rvalue) >= 0.5:
                findings.append({
                    "kind": "time_trend", "table": table, "columns": [dt, num],
                    "direction": "increasing" if lr.slope > 0 else "decreasing",
                    "r_value": round(float(lr.rvalue), 3),
                    "monthly_change_pct": round(float(rel_slope) * 100, 2),
                    "strength": round(abs(float(lr.rvalue)), 3),
                    "periods": int(len(monthly)),
                })

    findings.sort(key=lambda f: f["strength"], reverse=True)
    return findings[:40]
