"""Simple file-based storage for seen issues."""

import json
from pathlib import Path
from typing import Any, Dict, List


SEEN_PATH = Path("seen_issues.json")


def load_seen() -> List[int]:
    if not SEEN_PATH.exists():
        return []

    try:
        data = json.loads(SEEN_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [int(x) for x in data]
    except Exception:
        pass
    return []


def save_seen(issue_numbers: List[int]) -> None:
    SEEN_PATH.write_text(json.dumps(sorted(set(issue_numbers)), indent=2), encoding="utf-8")


def mark_seen(issue_numbers: List[int]) -> None:
    seen = set(load_seen())
    seen.update(issue_numbers)
    save_seen(sorted(seen))
