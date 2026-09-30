# Reproducibility

The repository exposes only the final SpliNet architecture: order-2, single-sided, 12 heads, radius 16. Historical order/sidedness sweeps are deliberately omitted.

## Shared tokenizer

All four model-training corpora use the same from-scratch 32k unigram SentencePiece tokenizer. It was trained on one million lowercased C4 text segments at the pinned C4 revision recorded in `tokenizer/tokenizer_metadata.json`.

The tokenizer SHA256 is:

`4837a155fbb73b166e8bc58c56a96334342e410af6ec888aeb31a0d620995b48`

Special IDs are fixed to `<unk>=0`, `<s>=1`, `</s>=2`, `<pad>=3`, `[CLS]=4`, `[SEP]=5`, `[MASK]=6`.

## Model initialization

The SpliNet/FNet comparison uses the same global seed (`2026`) and exact copying of all non-mixer modules from the same randomly initialized FNet scaffold. SpliNet's mixer has no trainable parameters.

The regular Transformer uses the same hidden size, layer count, FFN width, sequence length, vocabulary, tokenizer and training token budget, but includes learned self-attention projections and therefore has more trainable parameters.

## MLM protocol

The masking implementation is in `splinet/mlm.py`. It uses a deterministic mask seed, 15% eligible-token selection, an 80-token cap per sequence and the standard 80/10/10 mask/random/unchanged replacement rule.

## Training protocol

Adam uses learning rate `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay, and 1% linear warmup followed by constant learning rate. Dataset order is fixed by seed `7301`; MLM masks are fixed by seed `912431`.

The exact effective/micro-batch choices used for each recorded run are stored under `configs/training_*.json`.

## Checkpoint integrity

The 12 release archives are too large for normal GitHub source control. Their names, archive SHA256 values, extracted `.pt` SHA256 values and state-dict schemas are stored under `artifacts/`. Use `scripts/verify_checkpoint_archives.py` before attaching or consuming release assets.
