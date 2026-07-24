# DataAI — Architecture & Engineering Notes

**Live demo:** <https://iambatman07-ai-data-analyst.hf.space/> ·
**Source:** <https://github.com/Mark007-R/AI-Data-Analyst>

DataAI is a web app where a user uploads a dataset (CSV / XLSX / TSV — multi-sheet Excel
supported), a multi-stage pipeline analyzes it, and the app serves a page with:

1. **A written summary** explaining the data (LLM-generated, grounded in computed stats)
2. **Interactive charts** (ECharts) chosen automatically per column/relationship type
3. **A chatbot** that answers questions about the dataset by running **real SQL** through a
   **custom MCP server** — answers are computed, never guessed

---

## 1. Architecture

```
                        ┌──────────────────────────────────────────────┐
                        │                 Browser (SPA)                │
                        │  upload page → processing → dashboard + chat │
                        └───────┬──────────────────────────┬───────────┘
                                │ REST                     │ REST
                                ▼                          ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                          FastAPI backend (app/)                           │
│                                                                           │
│  POST /api/upload ──► ingest.py ──► DuckDB file per dataset (data/*.duckdb)│
│                          │                                                │
│                          ▼  (background task)                             │
│                 ┌─────────────────────┐                                   │
│                 │  Analysis pipeline  │   pipeline.py orchestrates:       │
│                 │  1 Profiler         │   column types, nulls, cardinality│
│                 │  2 Stats            │   Pearson/Spearman/Cramér's V/η²,  │
│                 │                     │   time trends                     │
│                 │  3 Viz              │   stats → ECharts option objects  │
│                 │  4 Report (LLM)     │   ranks insights + writes the     │
│                 │                     │   summary (structured output)     │
│                 └─────────────────────┘                                   │
│                                                                           │
│  POST /api/chat ──► chat.py ──► LLM tool-use loop ──► MCP client          │
│                                        │                    │ stdio       │
│                                        ▼                    ▼ (JSON-RPC)  │
│                                  tool_use blocks    ┌────────────────────┐│
│                                                     │ Custom MCP server  ││
│                                                     │ (mcp_server/)      ││
│                                                     │  • get_schema      ││
│                                                     │  • query (RO SQL)  ││
│                                                     │  • column_stats    ││
│                                                     │  • make_chart      ││
│                                                     │  • search_text     ││
│                                                     └─────────┬──────────┘│
│                                                               ▼           │
│                                                     DuckDB (read-only)    │
└───────────────────────────────────────────────────────────────────────────┘
```

**Key design decisions**

| Decision | Choice | Why |
|---|---|---|
| Analytics engine | **DuckDB**, one `.duckdb` file per upload | Every stage + the chatbot query the same engine, so answers and charts can never disagree; SQL over uploaded files is DuckDB's sweet spot |
| Tool protocol | **Custom MCP server** (FastMCP, stdio transport) | One ~250-line server exposes dataset tools with a `dataset_id` param — solves the one-DB-per-upload problem that off-the-shelf DuckDB MCP servers can't; no Node.js needed in the image |
| LLM | Structured outputs for the report (JSON always parses); tool-use loop for chat | The report call is constrained to a JSON schema; the chat loop lets the model query the data and self-correct SQL |
| Charts | **ECharts** rendered client-side from option JSON built server-side | Free, interactive, no license; a deterministic Viz stage builds options so the dashboard never depends on LLM availability |
| Chat → chart delivery | MCP `make_chart` saves the chart JSON to disk and returns a `chart_id`; the model embeds `[[chart:ID]]` in its answer; the frontend fetches `/api/chart/...` | Keeps big JSON out of the model's context (cheap + reliable) |
| Degraded mode | If `ANTHROPIC_API_KEY` is missing, the summary/insights fall back to template text from computed stats; chat is disabled with a clear message | Dashboard demo works with zero secrets |
| Deployment | Docker, port **7860** | Hugging Face Spaces convention |

---

## 2. The analysis pipeline

| Stage | File | LLM? | Job |
|---|---|---|---|
| **Ingest** | `app/ingest.py` | No | Reads CSV/TSV/XLSX (all sheets), sanitizes names, loads into a per-dataset DuckDB file, writes `meta.json` |
| **Profiler** | `app/profiler.py` | No | Classifies every column: `numeric` / `categorical` / `datetime` / `boolean` / `text` / `id`; computes nulls, cardinality, min/max/mean, top values, samples |
| **Stats** | `app/stats.py` | No | Type-aware pairwise relationships: numeric↔numeric → Pearson + Spearman; categorical↔numeric → correlation ratio (η²); categorical↔categorical → Cramér's V; datetime → monthly trend slope |
| **Viz** | `app/viz.py` | No | Turns profile + stats into ECharts options: histograms, top-category bars, time-series lines, scatter for top correlations, group-mean bars, correlation heatmap |
| **Report** | `app/report.py` | **Yes** | One LLM call with the full profile/stats JSON → structured output `{summary_markdown, insights[], data_quality_notes[]}`; ranks what's *interesting*, not just what's correlated. Falls back to a deterministic template report when no API key / on API error |
| **Chat** | `app/chat.py` | **Yes** | LLM tool-use loop with the MCP tools; system prompt carries the dataset schema so SQL is right on the first try |

## 3. The custom MCP server (`mcp_server/server.py`)

Built with **FastMCP** from the official `mcp` Python SDK. Transport: **stdio** (the backend
spawns it as a subprocess and holds one session for the app's lifetime).

| Tool | Signature | Notes |
|---|---|---|
| `get_schema` | `(dataset_id)` | Tables, columns, types, row counts |
| `query` | `(dataset_id, sql)` | **Read-only** DuckDB connection, SELECT/WITH only, results capped at 200 rows |
| `column_stats` | `(dataset_id, table, column)` | Quick describe for one column |
| `make_chart` | `(dataset_id, sql, chart_type, title, x_field, y_field)` | Runs the SQL, builds an ECharts option, saves it as `data/<id>.charts/<chart_id>.json`, returns the `chart_id` |
| `search_text` | `(dataset_id, table, column, term)` | Case-insensitive LIKE search for review-style text columns |

## 4. API surface

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | SPA (upload → processing → dashboard) |
| `/api/upload` | POST | Multipart file upload → returns `dataset_id`, starts background pipeline |
| `/api/status/{dataset_id}` | GET | `{stage, done, error}` for the processing screen |
| `/api/report/{dataset_id}` | GET | Summary markdown, insights, charts (ECharts options), schema |
| `/api/chat/{dataset_id}` | POST | `{message, history[]}` → `{answer, charts[]}` |
| `/api/chart/{dataset_id}/{chart_id}` | GET | Chart JSON saved by the MCP `make_chart` tool |
| `/api/health` | GET | `{ok, llm_enabled, mcp_ready}` |

## 5. Tech stack / dependencies

- **Python 3.11**, FastAPI + Uvicorn, python-multipart
- **pandas / numpy / scipy** — profiling & statistics
- **duckdb** — analytics engine
- **openpyxl** — XLSX reading; **xlrd** — legacy XLS reading
- **python-dotenv** — loads config from `.env`
- **anthropic[mcp]** — LLM SDK + MCP conversion helpers (`async_mcp_tool`)
- **mcp** — FastMCP server framework
- **ECharts 5**, **marked**, **DOMPurify** — self-hosted in `static/vendor/` (no CDN
  dependency; DOMPurify sanitizes all rendered markdown against XSS)

---

## 6. Security & robustness

The app treats uploaded data and all model-generated SQL as untrusted, and is built to
degrade rather than crash.

**SQL sandbox (MCP server).** The chatbot's `query`/`make_chart`/`column_stats`/`search_text`
tools run model-written SQL, so every DuckDB connection is opened read-only **and** with
`enable_external_access=false` + `lock_configuration=true`. This blocks filesystem and
network table functions (`read_csv`, `read_text`, `glob`, httpfs) and prevents re-enabling
them via `SET` — so SQL cannot read local files (e.g. the `.env`), reach other datasets, or
exfiltrate over the network. Table/column identifiers passed to the tools are validated
against the real schema and quote-escaped (no identifier injection); `dataset_id` is
whitelisted against `^[a-z0-9]{8,32}\Z` (no path traversal); non-SELECT statements are
rejected; results are capped in-query (`LIMIT`) with a 1 GB memory backstop so a runaway
join can't exhaust memory; and error text returned to the model is redacted for IO/internal
errors (SQL errors are kept so the model can self-correct).

**XSS.** Summary, quality notes, and chat answers are markdown rendered to HTML, so all
markdown output passes through **DOMPurify** before it reaches the DOM. The uploaded filename
is sanitized at ingest.

**Prompt injection.** The schema, filename, and profile values injected into the system
prompt are explicitly fenced as untrusted data.

**Input handling.** Upload size is checked before the body is buffered; `.xls`, corrupt
workbooks, mixed-timezone datetimes, colliding sheet/column names, single-column CSVs,
`inf`/all-null columns, and empty files are all handled without a 500.

**Runtime resilience.** Chat turns are bounded by a wall-clock timeout with typed handling
for rate-limit / overloaded / connection errors; the MCP session has a liveness probe and
read timeout; the status map is bounded; a failing chart never aborts the dashboard; and
transient LLM errors fall back to the template report (tagged so they're distinguishable
from the no-key case). Client-facing errors are generic; full detail goes to the logs.

## 7. Testing

- **Pipeline / edge cases:** a generated dataset exercises datetime, categorical, numeric,
  and free-text columns; a dedicated suite covers `inf`, all-null, single-column CSV,
  mixed-timezone dates, colliding Excel sheet names, and duplicate column names.
- **MCP server (real stdio JSON-RPC):** all five tools, plus negative tests for the SQL
  sandbox (filesystem/network/`SET` blocked), identifier injection, the row cap, and the
  `dataset_id` whitelist.
- **API layer (FastAPI TestClient):** health, upload, status polling, report, chat
  degradation, and the 400/404 validation guards.
- **UI:** verified in a browser — upload page, dashboard (summary, insight cards, ECharts,
  schema table), and chart sizing.

## 8. Run & deploy

```bash
cd DataAI
pip install -r requirements.txt      # or: .venv\Scripts\activate
set ANTHROPIC_API_KEY=...            # optional; enables the LLM summary + chatbot
uvicorn app.main:app --port 7860     # open http://localhost:7860
```

Deploy: ships as a Docker image (`Dockerfile`, port 7860) and runs as-is on a Hugging Face
Space with `sdk: docker`. Add `ANTHROPIC_API_KEY` as a Space secret to enable the LLM
features.

## 9. Known limitations / future work

- No native DuckDB query timeout (row caps + read-only + memory cap + external-access-off
  mitigate); a `con.interrupt()` watchdog is a possible future addition.
- Text/review analysis is LLM-sampled (no embedding RAG yet).
- Cross-sheet join inference (FK detection by value overlap) is scoped for a future version;
  each sheet/table is profiled separately today.
- Single-process deployment; per-dataset files make horizontal scaling straightforward later.
