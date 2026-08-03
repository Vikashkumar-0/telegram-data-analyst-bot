"""
data_tools.py
-------------
The actual "go get real data" building blocks. Kept separate from agent.py
so you can explain it on its own in an interview: "this module knows how to
fetch a file and load it into a DataFrame; the agent just decides when to
call it."

Everything downloaded lands in one shared SANDBOX_DIR so that a later
run_python_analysis() call can open the same file by path.
"""

import tempfile
import uuid
from pathlib import Path

import pandas as pd
import requests

SANDBOX_DIR = Path(tempfile.gettempdir()) / "databot_sandbox"
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

# Some government/public-data sites block requests with no User-Agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DataAnalystBot/1.0)"}


def download_file(url: str, timeout: int = 30) -> Path:
    """Downloads a URL into the shared sandbox directory, returns the local path."""
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    suffix = Path(url.split("?")[0]).suffix or ".bin"
    local_path = SANDBOX_DIR / f"{uuid.uuid4().hex}{suffix}"
    local_path.write_bytes(resp.content)
    return local_path


def load_tabular(path: Path) -> pd.DataFrame:
    """Best-effort loader for csv / tsv / xlsx / xls / json files."""
    suffix = Path(path).suffix.lower()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    # default: csv (also covers .csv and unknown-but-text extensions)
    return pd.read_csv(path)


def preview(df: pd.DataFrame, n: int = 5) -> dict:
    """A small, LLM-readable summary of a dataframe -- columns, shape, first rows."""
    return {
        "columns": list(df.columns),
        "shape": list(df.shape),
        "head": df.head(n).to_dict(orient="records"),
    }


def web_search(query: str, api_key: str, max_results: int = 5) -> list:
    """Free web search via Tavily (built for LLM agents -- returns clean
    extracted snippets, not raw SERP HTML). Imported lazily so the rest of
    this module has no hard dependency on the tavily package."""
    from tavily import TavilyClient

    tv = TavilyClient(api_key=api_key)
    resp = tv.search(query=query, max_results=max_results)
    results = []
    for r in resp.get("results", []):
        results.append(
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "snippet": (r.get("content") or "")[:500],
            }
        )
    return results
