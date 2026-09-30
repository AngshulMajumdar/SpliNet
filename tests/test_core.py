import torch

from splinet.core import inverse_kernel_values, order2_pole, order2_single_sided_headwise
from splinet.mixer import MultiHeadSpliNetMixer


def test_order2_pole_is_stable():
    assert -1.0 < order2_pole() < 0.0


def test_kernel_is_symmetric():
    k = inverse_kernel_values(16)
    assert len(k) == 33
    assert all(abs(a-b) < 1e-15 for a, b in zip(k, reversed(k)))


def test_headwise_shape():
    x = torch.randn(2, 4, 32, 8)
    y = order2_single_sided_headwise(x, radius=16)
    assert y.shape == x.shape


def test_mixer_has_no_trainable_parameters():
    mixer = MultiHeadSpliNetMixer(hidden_size=32, num_heads=4, radius=16)
    assert sum(p.numel() for p in mixer.parameters()) == 0
    x = torch.randn(2, 32, 32)
    assert mixer(x).shape == x.shape
