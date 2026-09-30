# Checkpoints

The source repository intentionally does not commit the trained checkpoint ZIP files because each archive is roughly 300–407 MB and exceeds GitHub's normal per-file source-control limit.

The complete release consists of twelve external checkpoint archives: SpliNet, FNet and the regular Transformer for FineWeb-Edu, arXiv, PG-19 and WikiText-103. The expected archive names and SHA256 hashes are in `artifacts/checkpoint_manifest.json`.

Recommended publication workflow: attach all twelve ZIP files as GitHub Release assets. Keep the repository itself small and cloneable.

To validate a directory containing the release assets:

```bash
python scripts/verify_checkpoint_archives.py /path/to/zips
```

To verify and extract one archive:

```bash
python scripts/extract_checkpoint_archive.py \
  /path/to/fineweb_edu_splinet_order2_single.zip \
  --output-dir checkpoints/fineweb_edu/splinet
```

The extractor checks both the ZIP hash and the extracted checkpoint hash before accepting the file.
