"""Pipeline orchestrator: runs the agent chain for one uploaded dataset and
persists the finished report JSON.

Stages (reported live to the frontend via /api/status):
    ingest -> profile -> stats -> charts -> report -> done
"""
from __future__ import annotations

import json
import traceback
from collections import OrderedDict

from . import config
from .ingest import load_meta, load_table
from .profiler import profile_table, columns_of_type
from .report import build_report
from .stats import analyze_table
from .viz import build_charts

# dataset_id -> {"stage": str, "done": bool, "error": str|None}. Bounded so a long-lived
# server can't accumulate status entries without limit; evicted "Done" entries are
# reconstructed from the report file by /api/status.
STATUS: "OrderedDict[str, dict]" = OrderedDict()
MAX_STATUS_ENTRIES = 500


def set_status(dataset_id: str, stage: str, done: bool = False, error: str | None = None):
    STATUS[dataset_id] = {"stage": stage, "done": done, "error": error}
    STATUS.move_to_end(dataset_id)  # keep in-flight runs from being evicted mid-poll
    while len(STATUS) > MAX_STATUS_ENTRIES:
        STATUS.popitem(last=False)


def run_pipeline(dataset_id: str) -> None:
    """Synchronous — executed in a worker thread by FastAPI's BackgroundTasks."""
    try:
        meta = load_meta(dataset_id)

        set_status(dataset_id, "Profiling columns")
        profiles, frames = [], {}
        for t in meta["tables"]:
            df = load_table(dataset_id, t["name"])
            frames[t["name"]] = df
            profiles.append(profile_table(df, t["name"]))

        set_status(dataset_id, "Finding relationships")
        findings: list[dict] = []
        for prof in profiles:
            findings.extend(analyze_table(frames[prof["table"]], prof))
        findings.sort(key=lambda f: f["strength"], reverse=True)

        set_status(dataset_id, "Building charts")
        charts: list[dict] = []
        for prof in profiles:
            table_findings = [f for f in findings if f["table"] == prof["table"]]
            charts.extend(build_charts(frames[prof["table"]], prof, table_findings))

        set_status(dataset_id, "Writing the summary (AI)")
        text_samples: dict[str, list[str]] = {}
        for prof in profiles:
            for col in columns_of_type(prof, "text"):
                s = frames[prof["table"]][col].dropna().astype(str)
                if len(s) > 0:
                    n = min(config.MAX_TEXT_SAMPLES_FOR_LLM, len(s))
                    text_samples[f"{prof['table']}.{col}"] = [
                        v[:300] for v in s.sample(n, random_state=7).tolist()]
        report = build_report(meta, profiles, findings, text_samples)

        payload = {
            "dataset_id": dataset_id,
            "meta": meta,
            "profiles": profiles,
            "findings": findings[:25],
            "charts": charts,
            "summary_markdown": report["summary_markdown"],
            "insights": report["insights"],
            "data_quality_notes": report.get("data_quality_notes", []),
            "report_source": report.get("source", "template"),
            "llm_error": report.get("llm_error"),  # set when a transient API error forced the template
            "llm_enabled": config.LLM_ENABLED,
        }
        config.dataset_report_path(dataset_id).write_text(
            json.dumps(payload, default=str), encoding="utf-8")
        set_status(dataset_id, "Done", done=True)
    except Exception:
        traceback.print_exc()  # full detail to server logs; generic text to the client
        set_status(dataset_id, "Failed", done=True, error="internal error — see server logs")
