---
title: AI Data Analyst
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# AI-Data-Analyst

**📦 Source:** [github.com/Mark007-R/AI-Data-Analyst](https://github.com/Mark007-R/AI-Data-Analyst)

Upload any CSV / TSV / Excel dataset and get:

- 📝 **A written summary** of what the data contains and what matters in it
- 📊 **Auto-generated interactive charts** (ECharts) picked per column/relationship type
- 💬 **A chatbot analyst** that answers questions by running **real SQL** against your
  data through a **custom MCP server** — answers are computed, never hallucinated

## How it works

1. **Ingest** — your file is loaded into a private per-upload DuckDB database
   (multi-sheet Excel: every sheet becomes a table)
2. **Multi-stage pipeline** — Profiler (column types & quality) → Stats
   (Pearson/Spearman, Cramér's V, η², time trends) → Viz (ECharts options) →
   Report (an LLM ranks the insights and writes the summary via structured outputs)
3. **Chat** — an LLM drives an agentic tool-use loop over a **custom FastMCP server**
   exposing `get_schema`, `query` (read-only SQL), `column_stats`, `make_chart`,
   `search_text`

The dashboard, charts and template summary work with **no API key**; a key enables the
LLM-written summary and the chatbot.

## Run locally

```bash
pip install -r requirements.txt
set ANTHROPIC_API_KEY=...              # optional — app degrades gracefully without it
uvicorn app.main:app --port 7860
# open http://localhost:7860
```

Without an API key, the dashboard/charts still work with a template summary; only the
chatbot and LLM-written summary are disabled.

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Enables the LLM summary + chatbot |
| `DATAAI_MODEL` | `claude-opus-4-8` | LLM model id used |
| `DATAAI_DATA_DIR` | `./data` | Where per-dataset DuckDB files live |

## Deploy

Ships as a Docker image (`Dockerfile`, port 7860) — deploys as-is to a Hugging Face
Space with `sdk: docker`. Set `ANTHROPIC_API_KEY` as a Space secret to enable the LLM
features.

