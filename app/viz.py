"""VizAgent: deterministic chart builder — profile + findings -> ECharts option objects.

The dashboard never depends on the LLM: charts are always computable from stats alone.
Each chart is {"id", "title", "reason", "option"} where option is a valid ECharts option.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .profiler import columns_of_type

PALETTE = ["#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de",
           "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc"]


def _base(title: str) -> dict:
    return {
        "color": PALETTE,
        "title": {"text": title, "left": "center", "textStyle": {"fontSize": 14}},
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 60, "right": 24, "top": 48, "bottom": 48, "containLabel": True},
    }


def _round_list(vals, sig=6):
    # Round to significant figures, not decimal places, so small-magnitude columns
    # (e.g. 0.00012) aren't flattened to all-zeros in the chart.
    out = []
    for v in vals:
        try:
            f = float(v)
            out.append(None if (np.isnan(f) or np.isinf(f)) else float(f"{f:.{sig}g}"))
        except (TypeError, ValueError):
            out.append(None)
    return out


def histogram(df: pd.DataFrame, col: str, table: str) -> dict | None:
    s = pd.to_numeric(df[col], errors="coerce")
    s = s[np.isfinite(s)]  # drop NaN AND +/-inf; np.histogram raises on inf ranges
    if len(s) < 10 or s.nunique() < 3:
        return None
    counts, edges = np.histogram(s, bins=min(30, max(8, int(np.sqrt(len(s))))))
    labels = [f"{edges[i]:.4g}–{edges[i+1]:.4g}" for i in range(len(counts))]
    opt = _base(f"Distribution of {col}")
    opt["tooltip"] = {"trigger": "axis"}
    opt["xAxis"] = {"type": "category", "data": labels, "axisLabel": {"rotate": 45, "fontSize": 10}}
    opt["yAxis"] = {"type": "value", "name": "count"}
    opt["series"] = [{"type": "bar", "data": [int(c) for c in counts], "barCategoryGap": "0%"}]
    return {"id": f"hist_{table}_{col}", "title": f"Distribution of {col}",
            "reason": f"Shows how values of the numeric column '{col}' are spread.", "option": opt}


def top_categories(df: pd.DataFrame, col: str, table: str) -> dict | None:
    vc = df[col].dropna().astype(str).value_counts().head(12)
    if len(vc) < 2:
        return None
    opt = _base(f"Top values of {col}")
    opt["xAxis"] = {"type": "value", "name": "count"}
    opt["yAxis"] = {"type": "category", "data": list(vc.index[::-1]),
                    "axisLabel": {"width": 120, "overflow": "truncate"}}
    opt["series"] = [{"type": "bar", "data": [int(v) for v in vc.values[::-1]]}]
    return {"id": f"topcat_{table}_{col}", "title": f"Top values of {col}",
            "reason": f"Most frequent categories in '{col}'.", "option": opt}


def time_series(df: pd.DataFrame, dt: str, num: str, table: str) -> dict | None:
    d = df[[dt, num]].dropna()
    if len(d) < 10:
        return None
    ts = d.set_index(dt)[num].sort_index()
    freq = "MS" if (ts.index.max() - ts.index.min()).days > 90 else "D"
    agg = ts.resample(freq).mean().dropna()
    if len(agg) < 3:
        return None
    fmt = "%Y-%m" if freq == "MS" else "%Y-%m-%d"
    opt = _base(f"{num} over time")
    opt["xAxis"] = {"type": "category", "data": [i.strftime(fmt) for i in agg.index]}
    opt["yAxis"] = {"type": "value", "name": num, "scale": True}
    opt["series"] = [{"type": "line", "data": _round_list(agg.values), "smooth": True,
                      "areaStyle": {"opacity": 0.15}}]
    return {"id": f"ts_{table}_{num}", "title": f"{num} over time",
            "reason": f"Average {num} per {'month' if freq == 'MS' else 'day'} based on '{dt}'.",
            "option": opt}


def scatter(df: pd.DataFrame, a: str, b: str, table: str, corr: float) -> dict | None:
    d = df[[a, b]].dropna()
    if len(d) < 10:
        return None
    if len(d) > config.CHART_POINT_CAP:
        d = d.sample(config.CHART_POINT_CAP, random_state=7)
    pts = [[x, y] for x, y in zip(_round_list(d[a].values), _round_list(d[b].values))
           if x is not None and y is not None]
    opt = _base(f"{a} vs {b} (r={corr:.2f})")
    opt["tooltip"] = {"trigger": "item"}
    opt["xAxis"] = {"type": "value", "name": a, "scale": True}
    opt["yAxis"] = {"type": "value", "name": b, "scale": True}
    opt["series"] = [{"type": "scatter", "data": pts, "symbolSize": 6,
                      "itemStyle": {"opacity": 0.6}}]
    return {"id": f"scatter_{table}_{a}_{b}", "title": f"{a} vs {b}",
            "reason": f"These two columns correlate (r={corr:.2f}).", "option": opt}


def group_means(df: pd.DataFrame, cat: str, num: str, table: str) -> dict | None:
    d = df[[cat, num]].dropna()
    if len(d) < 10:
        return None
    means = d.groupby(cat, observed=True)[num].mean().sort_values(ascending=False).head(12)
    if len(means) < 2:
        return None
    opt = _base(f"Average {num} by {cat}")
    opt["xAxis"] = {"type": "category", "data": [str(i) for i in means.index],
                    "axisLabel": {"rotate": 30, "fontSize": 10}}
    opt["yAxis"] = {"type": "value", "name": f"avg {num}"}
    opt["series"] = [{"type": "bar", "data": _round_list(means.values)}]
    return {"id": f"grp_{table}_{cat}_{num}", "title": f"Average {num} by {cat}",
            "reason": f"'{cat}' groups differ meaningfully on '{num}'.", "option": opt}


def corr_heatmap(df: pd.DataFrame, numeric: list[str], table: str) -> dict | None:
    numeric = numeric[: config.MAX_NUMERIC_COLS_FOR_PAIRS]
    if len(numeric) < 3:
        return None
    corr = df[numeric].corr(numeric_only=True)
    data = []
    for i, a in enumerate(numeric):
        for j, b in enumerate(numeric):
            v = corr.iloc[i, j]
            data.append([j, i, None if pd.isna(v) else round(float(v), 2)])
    opt = _base("Correlation matrix")
    opt["tooltip"] = {"position": "top"}
    opt["grid"] = {"left": 100, "right": 40, "top": 48, "bottom": 90, "containLabel": True}
    opt["xAxis"] = {"type": "category", "data": numeric, "axisLabel": {"rotate": 45, "fontSize": 10}}
    opt["yAxis"] = {"type": "category", "data": numeric, "axisLabel": {"fontSize": 10}}
    opt["visualMap"] = {"min": -1, "max": 1, "calculable": True, "orient": "horizontal",
                        "left": "center", "bottom": 0,
                        "inRange": {"color": ["#ee6666", "#ffffff", "#5470c6"]}}
    opt["series"] = [{"type": "heatmap", "data": data,
                      "label": {"show": len(numeric) <= 8, "fontSize": 9}}]
    return {"id": f"heat_{table}", "title": "Correlation matrix",
            "reason": "Pearson correlation between all numeric columns.", "option": opt}


def build_charts(df: pd.DataFrame, profile: dict, findings: list[dict]) -> list[dict]:
    table = profile["table"]
    charts: list[dict] = []
    seen: set[str] = set()

    def add(make):
        # Take a thunk, not a value: this isolates a single failing chart builder so
        # one bad column can't abort the whole dashboard (build errors are dropped).
        try:
            c = make()
        except Exception:
            return
        if c and c["id"] not in seen:
            seen.add(c["id"])
            charts.append(c)

    numeric = columns_of_type(profile, "numeric")
    categorical = columns_of_type(profile, "categorical")
    datetimes = columns_of_type(profile, "datetime")

    # 1) relationship charts for the strongest findings
    for f in findings[:6]:
        if f["kind"] == "numeric_numeric":
            a, b, p = f["columns"][0], f["columns"][1], f["pearson"]
            add(lambda a=a, b=b, p=p: scatter(df, a, b, table, p))
        elif f["kind"] == "categorical_numeric":
            a, b = f["columns"][0], f["columns"][1]
            add(lambda a=a, b=b: group_means(df, a, b, table))
        elif f["kind"] == "time_trend":
            a, b = f["columns"][0], f["columns"][1]
            add(lambda a=a, b=b: time_series(df, a, b, table))

    # 2) time series for primary datetime + top numerics
    if datetimes:
        for num in numeric[:2]:
            add(lambda num=num: time_series(df, datetimes[0], num, table))

    # 3) distributions and category breakdowns
    for col in numeric[:3]:
        add(lambda col=col: histogram(df, col, table))
    for col in categorical[:3]:
        add(lambda col=col: top_categories(df, col, table))

    # 4) heatmap
    add(lambda: corr_heatmap(df, numeric, table))

    return charts[:12]
