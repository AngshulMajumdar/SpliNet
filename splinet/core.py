import math

import torch
import torch.nn.functional as F


DEFAULT_RADIUS = 16
_KERNEL_CACHE = {}


def order2_pole():
    """Stable pole for the quadratic cardinal B-spline inverse prefilter."""
    return -3.0 + 2.0 * math.sqrt(2.0)


def inverse_kernel_values(radius=DEFAULT_RADIUS):
    """Truncated symmetric order-2 inverse-spline kernel.

    h[k] = sqrt(2) * z^|k|,  z = -3 + 2 sqrt(2),  |k| <= radius.
    """
    radius = int(radius)
    if radius < 1:
        raise ValueError("radius must be positive")
    z = order2_pole()
    return [math.sqrt(2.0) * z ** abs(k) for k in range(-radius, radius + 1)]


def _kernel(radius, dtype, device):
    key = (int(radius), dtype, device.type, device.index)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = torch.tensor(
            inverse_kernel_values(radius),
            dtype=dtype,
            device=device,
        ).reshape(1, 1, -1)
    return _KERNEL_CACHE[key]


def order2_single_sided_headwise(x, radius=DEFAULT_RADIUS):
    """Apply the selected SpliNet mixer.

    Parameters
    ----------
    x:
        Tensor of shape [batch, heads, sequence, head_dim].
    radius:
        Truncation radius of the fixed inverse B-spline filter.

    Returns
    -------
    Tensor with the same shape. The transform is applied only along the
    sequence axis, independently for every batch/head/feature signal.

    Notes
    -----
    The mixer has no trainable parameters and performs no cross-head mixing.
    """
    if x.ndim != 4:
        raise ValueError("Expected x with shape [B,H,L,Dh]")

    radius = int(radius)
    B, H, L, Dh = x.shape
    if L <= radius:
        raise ValueError(
            f"Sequence length {L} must exceed spline radius {radius}."
        )

    y = (
        x.permute(0, 1, 3, 2)
        .contiguous()
        .reshape(B * H * Dh, 1, L)
    )
    y = F.pad(y, (radius, radius), mode="reflect")
    y = F.conv1d(y, _kernel(radius, y.dtype, y.device))
    return (
        y.reshape(B, H, Dh, L)
        .permute(0, 1, 3, 2)
        .contiguous()
    )
