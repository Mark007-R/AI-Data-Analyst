"""Central configuration for DataAI."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root (ANTHROPIC_API_KEY etc.) before anything
# below reads os.environ. Real environment variables win over .env values.
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# Where per-dataset DuckDB files, metadata and charts live.
DATA_DIR = Path(os.environ.get("DATAAI_DATA_DIR", PROJECT_ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = PROJECT_ROOT / "static"
MCP_SERVER_PATH = PROJECT_ROOT / "mcp_server" / "server.py"

ANTHROPIC_MODEL = os.environ.get("DATAAI_MODEL", "claude-opus-4-8")
LLM_ENABLED = bool(os.environ.get("ANTHROPIC_API_KEY"))

# Analysis limits — keep pipelines fast and prompts bounded.
MAX_NUMERIC_COLS_FOR_PAIRS = 12      # cap pairwise correlation work
MAX_CATEGORICAL_CARDINALITY = 30     # cat columns above this are skipped for cat-cat tests
MAX_ROWS_SAMPLE = 50_000             # stats computed on a sample beyond this
MAX_TEXT_SAMPLES_FOR_LLM = 30        # review-style texts sent to the report call
QUERY_ROW_CAP = 200                  # rows returned by MCP query tool
CHART_POINT_CAP = 2000               # points in any single chart


def dataset_db_path(dataset_id: str) -> Path:
    return DATA_DIR / f"{dataset_id}.duckdb"


def dataset_meta_path(dataset_id: str) -> Path:
    return DATA_DIR / f"{dataset_id}.meta.json"


def dataset_report_path(dataset_id: str) -> Path:
    return DATA_DIR / f"{dataset_id}.report.json"


def dataset_charts_dir(dataset_id: str) -> Path:
    d = DATA_DIR / f"{dataset_id}.charts"
    d.mkdir(parents=True, exist_ok=True)
    return d
