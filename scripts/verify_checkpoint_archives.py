#!/usr/bin/env python3
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive_dir", type=Path)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "checkpoint_manifest.json",
    )
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    failed = False
    for item in manifest:
        path = args.archive_dir / item["archive"]
        if not path.exists():
            print("MISSING", path.name)
            failed = True
            continue
        digest = sha256(path)
        ok = digest == item["archive_sha256"]
        try:
            with zipfile.ZipFile(path, "r") as zf:
                bad = zf.testzip()
            ok = ok and bad is None
        except Exception:
            ok = False
        print(("OK     " if ok else "FAILED ") + path.name)
        failed |= not ok

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
