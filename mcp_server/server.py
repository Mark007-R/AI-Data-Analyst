"""DataAI custom MCP server (FastMCP, stdio transport).

Exposes the uploaded datasets as typed tools for the LLM:

    get_schema(dataset_id)                     -> tables, columns, types, row counts
    query(dataset_id, sql)                     -> read-only SQL, capped at 200 rows
    column_stats(dataset_id, table, column)    -> quick describe of one column
    make_chart(dataset_id, sql, chart_type, …) -> runs SQL, saves an ECharts option,
                                                  returns a chart_id for the frontend
    search_text(dataset_id, table, column, term) -> LIKE search in text columns

Guardrails: dataset_id whitelist regex (no path traversal), DuckDB opened read_only,
SELECT/WITH statements only, row caps everywhere, errors returned as friendly text
so the model can self-correct.

Run standalone for debugging:  python mcp_server/server.py
(Normally spawned by the backend over stdio.)
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import sys
from pathlib import Path

import duckdb
import pandas  # noqa: F401 — eager import: duckdb's .df() imports pandas lazily,
#               and a first-time pandas import inside a tool worker thread can
#               deadlock the stdio server. Import it in the main thread instead.
from mcp.server.fastmcp import FastMCP

DATA_DIR = Path(os.environ.get("DATAAI_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
ROW_CAP = int(os.environ.get("DATAAI_QUERY_ROW_CAP", "200"))
CHART_ROW_CAP = 500
DATASET_ID_RE = re.compile(r"^[a-z0-9]{8,32}\Z")  # \Z, not $, so a trailing newline can't sneak through

mcp = FastMCP("dataai-analyst")
_log = logging.getLogger("dataai-mcp")


# ---------------------------------------------------------------- helpers

def _db_path(dataset_id: str) -> Path:
    if not DATASET_ID_RE.match(dataset_id or ""):
        raise ValueError(f"Invalid dataset_id: {dataset_id!r}")
    p = DATA_DIR / f"{dataset_id}.duckdb"
    if not p.exists():
        raise ValueError(f"No dataset found with id {dataset_id!r}")
    return p


def _connect(dataset_id: str) -> duckdb.DuckDBPyConnection:
    # enable_external_access=false blocks read_csv/read_parquet/read_text/glob and httpfs
    # (no filesystem or network reads → can't exfiltrate the .env key or other datasets);
    # lock_configuration=true stops the model re-enabling it via SET; memory_limit caps
    # intermediate materialization that LIMIT push-down can't bound.
    return duckdb.connect(
        str(_db_path(dataset_id)), read_only=True,
        config={"enable_external_access": "false",
                "lock_configuration": "true",
                "memory_limit": "1GB"})


def _safe_table(con: duckdb.DuckDBPyConnection, table: str) -> str:
    """Validate a table name against the real schema and return it safely quoted.
    Prevents identifier injection through the table parameter of column_stats/search_text."""
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if table not in tables:
        raise ValueError(f"Table {table!r} not found. Available: {sorted(tables)[:40]}")
    return '"' + table.replace('"', '""') + '"'


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _check_select_only(sql: str) -> tuple[str, str]:
    """Return (sanitized_sql, leading_keyword). Rejects anything but one read-only statement."""
    stripped = re.sub(r"--[^\n]*", " ", sql)          # strip line comments
    stripped = re.sub(r"/\*.*?\*/", " ", stripped, flags=re.S).strip().rstrip(";").strip()
    if ";" in stripped:
        raise ValueError("Multiple statements are not allowed — send one SELECT.")
    first = stripped.lstrip("(").split(None, 1)
    if not first or first[0].lower() not in ("select", "with", "describe", "show"):
        raise ValueError("Only read-only SELECT/WITH queries are allowed.")
    return stripped, first[0].lower()


def _df_to_result(df, cap: int) -> str:
    total = len(df)
    if total > cap:
        df = df.head(cap)
    payload = {
        "row_count_returned": int(len(df)),
        "row_count_total": None if total > cap else int(total),  # unknown when capped in-query
        "truncated": bool(total > cap),
        "columns": list(map(str, df.columns)),
        "rows": json.loads(df.to_json(orient="values", date_format="iso")),
    }
    return json.dumps(payload, default=str)


def _err(e: Exception) -> str:
    # Echo SQL/schema errors so the model can self-correct; redact IO/internal errors
    # (they can leak host paths and act as a filesystem oracle).
    safe = (duckdb.ParserException, duckdb.BinderException,
            duckdb.CatalogException, duckdb.ConversionException,
            duckdb.InvalidInputException, ValueError)
    if isinstance(e, safe):
        msg = str(e)
    else:
        _log.warning("dataai-mcp query error: %s", e)
        msg = "Query failed — check the table/column names, or it hit an internal error."
    return json.dumps({"error": msg,
                       "hint": "Fix the SQL and try again. Use get_schema to check table/column names."})


# ---------------------------------------------------------------- tools

@mcp.tool()
def get_schema(dataset_id: str) -> str:
    """List every table in the dataset with its columns, column types and row count.
    Always call this before writing SQL if you are unsure of exact names."""
    try:
        con = _connect(dataset_id)
        try:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            out = []
            for t in tables:
                cols = con.execute(f'DESCRIBE "{t}"').fetchall()
                rows = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                out.append({
                    "table": t,
                    "rows": int(rows),
                    "columns": [{"name": c[0], "type": c[1]} for c in cols],
                })
            return json.dumps({"tables": out})
        finally:
            con.close()
    except Exception as e:
        return _err(e)


@mcp.tool()
def query(dataset_id: str, sql: str) -> str:
    """Run a read-only SQL query (DuckDB dialect) against the dataset and return the
    resulting rows as JSON. Only SELECT/WITH statements are allowed; results are capped
    at 200 rows, so aggregate rather than dumping raw tables. Quote identifiers with
    double quotes if they contain unusual characters."""
    try:
        sql, kind = _check_select_only(sql)
        if kind in ("select", "with"):
            # Cap rows in-query so .df() never materializes a giant result set.
            sql = f"SELECT * FROM ({sql}) AS _q LIMIT {ROW_CAP + 1}"
        con = _connect(dataset_id)
        try:
            df = con.execute(sql).df()
        finally:
            con.close()
        return _df_to_result(df, ROW_CAP)
    except Exception as e:
        return _err(e)


@mcp.tool()
def column_stats(dataset_id: str, table: str, column: str) -> str:
    """Quick statistics for one column: count, nulls, distinct values, min/max/avg for
    numerics, and the 10 most frequent values."""
    try:
        con = _connect(dataset_id)
        try:
            tq = _safe_table(con, table)
            cols = {c[0]: c[1] for c in con.execute(f'DESCRIBE {tq}').fetchall()}
            if column not in cols:
                raise ValueError(f"Column {column!r} not found in table {table!r}. "
                                 f"Available: {list(cols)[:40]}")
            q = _quote_ident(column)
            base = con.execute(
                f'SELECT COUNT(*) AS total, COUNT({q}) AS non_null, COUNT(DISTINCT {q}) AS distinct_vals '
                f'FROM {tq}').fetchone()
            result = {"table": table, "column": column, "type": cols[column],
                      "total_rows": int(base[0]), "non_null": int(base[1]),
                      "distinct": int(base[2])}
            if any(k in cols[column].upper() for k in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "HUGEINT")):
                mn, mx, avg, med = con.execute(
                    f'SELECT MIN({q}), MAX({q}), AVG({q}), MEDIAN({q}) FROM {tq}').fetchone()
                result.update({"min": mn, "max": mx, "avg": avg, "median": med})
            top = con.execute(
                f'SELECT CAST({q} AS VARCHAR) AS v, COUNT(*) AS n FROM {tq} '
                f'WHERE {q} IS NOT NULL GROUP BY 1 ORDER BY n DESC LIMIT 10').fetchall()
            result["top_values"] = [{"value": v, "count": int(n)} for v, n in top]
            return json.dumps(result, default=str)
        finally:
            con.close()
    except Exception as e:
        return _err(e)


@mcp.tool()
def make_chart(dataset_id: str, sql: str, chart_type: str, title: str,
               x_field: str, y_field: str) -> str:
    """Create a chart for the user from a SQL query and return its chart_id.

    chart_type must be one of: bar, line, pie, scatter.
    The SQL must return the columns named by x_field and y_field (aggregate first —
    at most 500 rows are charted). After this tool succeeds, include the marker
    [[chart:CHART_ID]] on its own line in your answer, where CHART_ID is the id
    returned by this tool — the user interface renders the chart at that marker."""
    try:
        chart_type = chart_type.strip().lower()
        if chart_type not in ("bar", "line", "pie", "scatter"):
            raise ValueError("chart_type must be one of: bar, line, pie, scatter")
        sql, kind = _check_select_only(sql)
        if kind in ("select", "with"):
            sql = f"SELECT * FROM ({sql}) AS _q LIMIT {CHART_ROW_CAP + 1}"
        con = _connect(dataset_id)
        try:
            df = con.execute(sql).df()
        finally:
            con.close()
        if x_field not in df.columns or y_field not in df.columns:
            raise ValueError(f"SQL result columns are {list(df.columns)}; "
                             f"expected x_field={x_field!r} and y_field={y_field!r}.")
        if len(df) == 0:
            raise ValueError("The SQL returned no rows — nothing to chart.")
        df = df.head(CHART_ROW_CAP)

        x = [str(v) for v in df[x_field].tolist()]
        y = []
        for v in df[y_field].tolist():
            try:
                y.append(round(float(v), 4))
            except (TypeError, ValueError):
                y.append(None)

        option: dict = {
            "title": {"text": title, "left": "center", "textStyle": {"fontSize": 14}},
            "tooltip": {"trigger": "item" if chart_type in ("pie", "scatter") else "axis"},
            "grid": {"left": 60, "right": 24, "top": 48, "bottom": 60, "containLabel": True},
        }
        if chart_type == "pie":
            option["series"] = [{
                "type": "pie", "radius": ["30%", "65%"],
                "data": [{"name": n, "value": v} for n, v in zip(x, y) if v is not None],
                "label": {"formatter": "{b}: {c} ({d}%)"},
            }]
        elif chart_type == "scatter":
            xs = []
            for v in df[x_field].tolist():
                try:
                    xs.append(round(float(v), 4))
                except (TypeError, ValueError):
                    xs.append(None)
            option["xAxis"] = {"type": "value", "name": x_field, "scale": True}
            option["yAxis"] = {"type": "value", "name": y_field, "scale": True}
            option["series"] = [{"type": "scatter", "symbolSize": 7,
                                 "data": [[a, b] for a, b in zip(xs, y)
                                          if a is not None and b is not None]}]
        else:
            option["xAxis"] = {"type": "category", "data": x,
                               "axisLabel": {"rotate": 30, "fontSize": 10}}
            option["yAxis"] = {"type": "value", "name": y_field}
            option["series"] = [{"type": chart_type, "data": y,
                                 **({"smooth": True} if chart_type == "line" else {})}]

        chart_id = f"c{secrets.token_hex(6)}"
        charts_dir = DATA_DIR / f"{dataset_id}.charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        (charts_dir / f"{chart_id}.json").write_text(
            json.dumps({"id": chart_id, "title": title, "option": option}), encoding="utf-8")
        return json.dumps({
            "chart_id": chart_id, "points": int(len(df)),
            "instruction": f"Chart saved. Put the marker [[chart:{chart_id}]] on its own "
                           f"line in your answer where the chart should appear.",
        })
    except Exception as e:
        return _err(e)


@mcp.tool()
def search_text(dataset_id: str, table: str, column: str, term: str) -> str:
    """Case-insensitive substring search inside a text column (e.g. reviews or comments).
    Returns up to 20 matching rows with the matched text. Use this to find and quote
    what people actually wrote about a topic."""
    try:
        if not term or len(term) > 200:
            raise ValueError("Provide a search term between 1 and 200 characters.")
        con = _connect(dataset_id)
        try:
            tq = _safe_table(con, table)
            cols = {c[0] for c in con.execute(f'DESCRIBE {tq}').fetchall()}
            if column not in cols:
                raise ValueError(f"Column {column!r} not found in table {table!r}.")
            cq = _quote_ident(column)
            df = con.execute(
                f'SELECT * FROM {tq} WHERE CAST({cq} AS VARCHAR) ILIKE ? LIMIT 20',
                [f"%{term}%"]).df()
            total = con.execute(
                f'SELECT COUNT(*) FROM {tq} WHERE CAST({cq} AS VARCHAR) ILIKE ?',
                [f"%{term}%"]).fetchone()[0]
        finally:
            con.close()
        payload = json.loads(_df_to_result(df, 20))
        payload["total_matches"] = int(total)
        return json.dumps(payload, default=str)
    except Exception as e:
        return _err(e)


if __name__ == "__main__":
    print(f"[dataai-mcp] serving datasets from {DATA_DIR}", file=sys.stderr)
    mcp.run()  # stdio transport
