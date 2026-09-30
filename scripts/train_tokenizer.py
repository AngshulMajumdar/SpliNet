#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path

import sentencepiece as spm
from datasets import load_dataset

C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
VOCAB_SIZE = 32000
TRAIN_SEGMENTS = 1_000_000
MAX_SEGMENT_CHARS = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, default=Path("/tmp/splinet_tokenizer"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    corpus = args.work_dir / "c4_sentencepiece_corpus.txt"
    prefix = args.work_dir / "spiece"

    stream = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
        revision=C4_REVISION,
    )

    segments = 0
    docs = 0
    with corpus.open("w", encoding="utf-8") as f:
        for example in stream:
            docs += 1
            text = (example.get("text") or "").lower()
            for segment in text.splitlines():
                segment = " ".join(segment.split())
                if not segment:
                    continue
                if len(segment) > MAX_SEGMENT_CHARS:
                    segment = segment[:MAX_SEGMENT_CHARS]
                f.write(segment + "\n")
                segments += 1
                if segments >= TRAIN_SEGMENTS:
                    break
            if segments >= TRAIN_SEGMENTS:
                break

    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        model_type="unigram",
        vocab_size=VOCAB_SIZE,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
        user_defined_symbols=["[CLS]", "[SEP]", "[MASK]"],
        character_coverage=1.0,
        normalization_rule_name="nmt_nfkc",
        input_sentence_size=TRAIN_SEGMENTS,
        shuffle_input_sentence=True,
        hard_vocab_limit=True,
        num_threads=max(1, min(16, os.cpu_count() or 4)),
        max_sentence_length=4096,
    )

    model_out = args.output_dir / "spiece.model"
    vocab_out = args.output_dir / "spiece.vocab"
    model_out.write_bytes((prefix.with_suffix(".model")).read_bytes())
    vocab_out.write_bytes((prefix.with_suffix(".vocab")).read_bytes())
    sha = hashlib.sha256(model_out.read_bytes()).hexdigest()

    meta = {
        "name": "SpliNet shared tokenizer",
        "trained_from_scratch": True,
        "pretrained_tokenizer": False,
        "source": "allenai/c4",
        "source_revision": C4_REVISION,
        "training_segments": TRAIN_SEGMENTS,
        "source_documents_consumed": docs,
        "lowercase": True,
        "model_type": "unigram",
        "vocab_size": VOCAB_SIZE,
        "special_token_ids": {
            "<unk>": 0,
            "<s>": 1,
            "</s>": 2,
            "<pad>": 3,
            "[CLS]": 4,
            "[SEP]": 5,
            "[MASK]": 6,
        },
        "model_sha256": sha,
    }
    (args.output_dir / "tokenizer_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
