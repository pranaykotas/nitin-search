from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Record:
    source: str
    title: str
    reference: str
    date: str
    text: str


def write_records(records: list[Record], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in records], f)


def read_records(path: str) -> list[Record]:
    with open(path) as f:
        raw = json.load(f)
    return [Record(**r) for r in raw]
