import json
from pathlib import Path


def test_manifest_contains_exact_release_set():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "artifacts" / "checkpoint_manifest.json").read_text())
    assert len(data) == 12
    assert {x["model"] for x in data} == {"SpliNet", "FNet", "Transformer"}
    assert {x["dataset"] for x in data} == {"fineweb_edu", "arxiv", "pg19", "wikitext103"}
