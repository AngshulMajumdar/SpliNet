import torch.nn as nn

from .core import order2_single_sided_headwise


class MultiHeadSpliNetMixer(nn.Module):
    """Fixed zero-parameter multi-head order-2 single-sided spline mixer."""

    def __init__(self, hidden_size, num_heads=12, radius=16):
        super().__init__()
        hidden_size = int(hidden_size)
        num_heads = int(num_heads)
        radius = int(radius)

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.radius = radius

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Expected x with shape [B,L,D]")

        B, L, D = x.shape
        if D != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, received {D}"
            )

        y = (
            x.reshape(B, L, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        y = order2_single_sided_headwise(y, radius=self.radius)
        return (
            y.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(B, L, D)
        )

    def extra_repr(self):
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            "order=2, sidedness=single, "
            f"radius={self.radius}, parameters=0"
        )
