"""Ingestion: uploaded file -> per-dataset DuckDB database + meta.json.

Supports CSV / TSV / TXT (delimiter-sniffed) and XLSX/XLS (every sheet becomes a table).
"""
from __future__ import annotations

import csv
import io
import json
import re
import secrets
import time

import duckdb
import pandas as pd

from . import config


class IngestError(Exception):
    pass


def new_dataset_id() -> str:
    return secrets.token_hex(8)  # 16 lowercase hex chars, matches ^[a-z0-9]{8,32}$


def _sanitize_name(name: str, fallback: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name).strip()).strip("_").lower()
    if not name:
        return fallback
    if name[0].isdigit():
        name = f"t_{name}"
    return name[:60]


def _dedupe_name(name: str, used: set[str]) -> str:
    """Return `name`, or the first free `name_N`, given the set of already-taken names."""
    if name not in used:
        return name
    n = 1
    while f"{name}_{n}" in used:
        n += 1
    return f"{name}_{n}"


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Probe against every assigned name so the suffixed candidate is itself free —
    # a plain counter can still collide (e.g. 'a','a 1','a!' -> a, a_1, a_1).
    used: set[str] = set()
    cols = []
    for i, c in enumerate(df.columns):
        c2 = _dedupe_name(_sanitize_name(c, f"col_{i}"), used)
        used.add(c2)
        cols.append(c2)
    df.columns = cols
    return df


def _read_tabular(raw: bytes, filename: str) -> dict[str, pd.DataFrame]:
    """Return {table_name: DataFrame} for the uploaded file."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext in ("xlsx", "xls", "xlsm"):
        try:
            sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
        except Exception as e:  # bad zip, missing xls engine, corrupt workbook
            raise IngestError(f"Could not read the Excel file: {e}")
        out: dict[str, pd.DataFrame] = {}
        for i, (sheet, df) in enumerate(sheets.items()):
            if df is None or df.empty:
                continue
            # Distinct sheets can sanitize to the same name — dedupe so none is dropped.
            name = _dedupe_name(_sanitize_name(sheet, f"sheet_{i}"), set(out))
            out[name] = df
        if not out:
            raise IngestError("The Excel file contains no non-empty sheets.")
        return out

    if ext in ("csv", "tsv", "txt", ""):
        last_err: Exception | None = None
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                text = raw.decode(enc)
                if ext == "tsv":
                    sep = "\t"
                else:
                    # Restrict sniffing to real delimiters; the unrestricted sniffer
                    # picks an arbitrary letter on single-column files instead of failing.
                    sample = "\n".join(text.splitlines()[:25])
                    try:
                        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
                    except csv.Error:
                        sep = ","  # no delimiter found: single-column file, comma is safe
                df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
                if df.empty or len(df.columns) == 0:
                    raise IngestError("The file parsed but contains no data.")
                base = _sanitize_name(filename.rsplit(".", 1)[0], "dataset")
                return {base: df}
            except IngestError:
                raise
            except Exception as e:  # try next encoding
                last_err = e
        raise IngestError(f"Could not parse the file as CSV/TSV: {last_err}")

    raise IngestError(f"Unsupported file type: .{ext} (use CSV, TSV or XLSX)")


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleanup: parse obvious datetime columns, strip stringly whitespace."""
    for col in df.columns:
        s = df[col]
        # pandas >= 3 uses the dedicated `str` dtype; older versions use object
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            s = s.astype(str).str.strip().replace({"": None, "nan": None, "None": None})
            df[col] = s
            # datetime detection on a sample: only convert when it parses cleanly
            sample = s.dropna().head(200)
            if len(sample) >= 5:
                looks_datey = sample.str.contains(r"\d{1,4}[-/ ]\w{1,9}[-/ ]\d{1,4}", regex=True).mean() > 0.8
                if looks_datey:
                    try:
                        parsed = pd.to_datetime(s, errors="coerce", format="mixed")
                    except (ValueError, TypeError):
                        # pandas 3 raises on mixed UTC offsets even with errors="coerce"
                        parsed = pd.to_datetime(s, errors="coerce", format="mixed", utc=True)
                    if parsed.notna().sum() >= 0.85 * s.notna().sum():
                        df[col] = parsed
    return df


def ingest(raw: bytes, filename: str) -> dict:
    """Load the uploaded file into a fresh DuckDB database. Returns the meta dict."""
    dataset_id = new_dataset_id()
    frames = _read_tabular(raw, filename)

    db_path = config.dataset_db_path(dataset_id)
    con = duckdb.connect(str(db_path))
    tables = []
    try:
        for name, df in frames.items():
            df = _sanitize_columns(df)
            df = _coerce_types(df)
            con.register("df_in", df)
            con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM df_in')
            con.unregister("df_in")
            tables.append({"name": name, "rows": int(len(df)), "columns": list(df.columns)})
    except Exception:
        con.close()
        db_path.unlink(missing_ok=True)  # don't leave an orphaned half-built db on disk
        raise
    finally:
        con.close()  # idempotent — harmless second close on the success path

    # Store a sanitized, length-capped filename: it flows into the LLM system prompt
    # and into markdown rendered as HTML, so strip control chars and bound its length.
    safe_filename = re.sub(r"[\x00-\x1f\x7f]", "", str(filename)).strip()[:120] or "upload"
    meta = {
        "dataset_id": dataset_id,
        "filename": safe_filename,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tables": tables,
    }
    config.dataset_meta_path(dataset_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_meta(dataset_id: str) -> dict:
    p = config.dataset_meta_path(dataset_id)
    if not p.exists():
        raise IngestError(f"Unknown dataset: {dataset_id}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_table(dataset_id: str, table: str) -> pd.DataFrame:
    con = duckdb.connect(str(config.dataset_db_path(dataset_id)), read_only=True)
    try:
        df = con.execute(f'SELECT * FROM "{table}"').df()
    finally:
        con.close()
    total = len(df)
    if total > config.MAX_ROWS_SAMPLE:
        df = df.sample(config.MAX_ROWS_SAMPLE, random_state=7)
    df.attrs["total_rows"] = total  # so the profiler reports true size, not the sample size
    return df
