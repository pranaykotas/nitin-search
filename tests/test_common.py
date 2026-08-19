import json

from ingest.common import Record, read_records, write_records


def test_write_then_read_round_trips(tmp_path):
    records = [
        Record(source="blog", title="A Post", reference="https://example.com/a", date="2024-01-01", text="Hello world"),
        Record(source="notes", title="A Note", reference="notes/a.md", date="2024-01-02", text="Some thoughts"),
    ]
    out_path = str(tmp_path / "records.json")

    write_records(records, out_path)
    loaded = read_records(out_path)

    assert loaded == records


def test_write_records_creates_parent_directories(tmp_path):
    out_path = str(tmp_path / "nested" / "dir" / "records.json")
    write_records([Record(source="blog", title="X", reference="x", date="", text="y")], out_path)

    with open(out_path) as f:
        data = json.load(f)
    assert len(data) == 1
