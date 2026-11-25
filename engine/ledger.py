"""Append-only JSONL files that survive concurrency and crashes, and the API cost ledger built on them.

Each record is one line written by a single os.write on an O_APPEND descriptor while holding an exclusive
flock, then fsync'ed. Concurrent writers (threads or processes) therefore never interleave bytes within a
line, and a line that returned from append_jsonl is on disk. A crash can at worst leave one partial trailing
line, which the readers skip.
"""

import fcntl
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "budget" / "ledger.jsonl"


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            n = os.write(fd, data)
            if n != len(data):  # regular-file writes this small do not short-write; refuse to leave half a line
                raise OSError(f"short write to {path}: {n}/{len(data)} bytes")
            os.fsync(fd)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def read_jsonl(path: Path):
    """Yield parsed records, skipping blank or partially written lines."""
    path = Path(path)
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def record_call(experiment_id: str, model: str, batch: bool, tokens: dict, cost_usd: float,
                request_id: str | None, path: Path = LEDGER) -> dict:
    """Append one API call to the ledger. `tokens` is engine.pricing.usage_tokens(usage)."""
    rec = {
        "ts": time.time(),
        "experiment_id": experiment_id,
        "model": model,
        "batch": bool(batch),
        "input_tokens": tokens["input_tokens"],
        "cache_write_tokens": tokens["cache_write_tokens"],
        "cache_read_tokens": tokens["cache_read_tokens"],
        "output_tokens": tokens["output_tokens"],
        "cost_usd": cost_usd,
        "request_id": request_id,
    }
    append_jsonl(path, rec)
    return rec


def total_api_spend(path: Path = LEDGER, experiment_id: str | None = None) -> float:
    """Sum of cost_usd over the ledger (optionally for one experiment). Also used by scripts/spend_watch.py."""
    total = 0.0
    for rec in read_jsonl(path):
        if experiment_id is not None and rec.get("experiment_id") != experiment_id:
            continue
        try:
            total += float(rec.get("cost_usd", 0.0))
        except (TypeError, ValueError):
            continue
    return total
