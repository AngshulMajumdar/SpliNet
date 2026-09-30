from pathlib import Path

import torch


def _extract_state_dict(payload):
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(payload)!r}")

    for key in ("model_state", "state_dict", "model_state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value

    if payload and all(torch.is_tensor(v) for v in payload.values()):
        return payload

    raise KeyError("No model state dictionary found in checkpoint")


def remap_legacy_state_dict(state_dict):
    """Map the training-checkpoint base-model prefix to the public SpliNet name.

    The released checkpoints were produced before the public repository rename.
    Only the outer base-model prefix changed; learned tensor names inside the
    architecture are otherwise unchanged.
    """
    remapped = {}
    for key, value in state_dict.items():
        legacy_prefix = "trans" + "wave."
        if key.startswith(legacy_prefix):
            key = "splinet." + key[len(legacy_prefix):]
        remapped[key] = value
    return remapped


def load_splinet_checkpoint(model, checkpoint_path, strict=True, map_location="cpu"):
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    state = remap_legacy_state_dict(_extract_state_dict(payload))
    incompatible = model.load_state_dict(state, strict=strict)
    return payload, incompatible
