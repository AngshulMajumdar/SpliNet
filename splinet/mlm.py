import numpy as np
import torch

VOCAB_SIZE = 32000
SPECIAL_IDS = (0, 1, 2, 3, 4, 5, 6)
MASK_ID = 6
MLM_PROBABILITY = 0.15
MAX_PREDICTIONS = 80
MASK_SEED = 912431


def make_mlm_batch(clean_np, seed_index, validation=False):
    """Deterministic 15% MLM corruption with an 80-token cap and 80/10/10 replacement."""
    clean = torch.from_numpy(np.asarray(clean_np, dtype=np.int64).copy())
    ids = clean.clone()
    labels = torch.full_like(ids, -100)

    seed = MASK_SEED + int(seed_index)
    if validation:
        seed += 1_000_000_000
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    eligible = torch.ones_like(ids, dtype=torch.bool)
    for sid in SPECIAL_IDS:
        eligible &= ids.ne(sid)

    scores = torch.rand(ids.shape, generator=g)
    selected = eligible & scores.lt(MLM_PROBABILITY)

    ranked_scores = scores.masked_fill(~selected, float("inf"))
    k = min(MAX_PREDICTIONS, ids.shape[1])
    keep_index = torch.topk(
        ranked_scores,
        k=k,
        dim=1,
        largest=False,
        sorted=False,
    ).indices
    keep = torch.zeros_like(selected)
    keep.scatter_(1, keep_index, True)
    selected &= keep

    labels[selected] = ids[selected]
    replacement = torch.rand(ids.shape, generator=g)
    mask_positions = selected & replacement.lt(0.80)
    random_positions = selected & replacement.ge(0.80) & replacement.lt(0.90)

    ids[mask_positions] = MASK_ID
    random_ids = torch.randint(7, VOCAB_SIZE, ids.shape, generator=g, dtype=torch.long)
    ids[random_positions] = random_ids[random_positions]
    return ids, labels
