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
    ap.add_argument("archive", type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "checkpoint_manifest.json",
    )
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rec = next((x for x in manifest if x["archive"] == args.archive.name), None)
    if rec is None:
        raise SystemExit(f"Archive not listed in manifest: {args.archive.name}")
    digest = sha256(args.archive)
    if digest != rec["archive_sha256"]:
        raise SystemExit("Archive SHA256 mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise SystemExit(f"Corrupt archive member: {bad}")
        zf.extractall(args.output_dir)
    extracted = args.output_dir / "trained_model_final.pt"
    if sha256(extracted) != rec["checkpoint_sha256"]:
        raise SystemExit("Extracted checkpoint SHA256 mismatch")
    print(extracted)


if __name__ == "__main__":
    main()
