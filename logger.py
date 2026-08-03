"""
logger.py
---------
Writes one JSON line per event, straight to a file on disk. No GitHub API,
no external service -- bot.py serves this same file over HTTP, so the
log_url just points back at this app. Simpler and more reliable than
committing to a repo on every message.

Each chat gets its own file: logs/<chat_id>.jsonl. Lines are appended as
they happen (not batched), so the log is readable even while a run is still
in progress.
"""

import json
import time
from pathlib import Path

import config

LOG_DIR = Path(config.LOG_DIR)
LOG_DIR.mkdir(parents=True, exist_ok=True)


class RunLogger:
    def __init__(self, chat_id):
        self.chat_id = chat_id
        self.path = LOG_DIR / f"{chat_id}.jsonl"

    def log(self, entry: dict):
        entry = dict(entry)
        entry["ts"] = time.time()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    @property
    def url(self) -> str:
        return f"{config.PUBLIC_BASE_URL}/logs/{self.chat_id}.jsonl"
