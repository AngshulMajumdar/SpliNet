import random

import numpy as np
import torch
from transformers import FNetForMaskedLM

from .configuration_splinet import SpliNetConfig
from .modeling_splinet import SpliNetForMaskedLM
from baselines.fnet import base_fnet_config


def set_all_seeds(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _copy_module_exact(source, destination):
    destination.load_state_dict(source.state_dict(), strict=True)


def base_splinet_config():
    return SpliNetConfig(
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=12,
        intermediate_size=3072,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=4,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=3,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        splinet_num_heads=12,
        splinet_radius=16,
    )


def build_splinet(seed=2026, matched_to_fnet=True):
    """Build the final SpliNet architecture.

    With matched_to_fnet=True, copy every non-mixer parameter from an FNet
    initialized with the same seed, exactly matching the comparison protocol.
    """
    set_all_seeds(seed)
    cfg = base_splinet_config()
    model = SpliNetForMaskedLM(cfg)

    if not matched_to_fnet:
        return model

    set_all_seeds(seed)
    fcfg = base_fnet_config()
    fcfg.use_tpu_fourier_optimizations = False
    fnet = FNetForMaskedLM(fcfg)
    fnet.fnet.pooler = None

    _copy_module_exact(fnet.fnet.embeddings, model.splinet.embeddings)
    for i in range(cfg.num_hidden_layers):
        f = fnet.fnet.encoder.layer[i]
        s = model.splinet.encoder.layer[i]
        _copy_module_exact(f.fourier.output.LayerNorm, s.mixing.LayerNorm)
        _copy_module_exact(f.intermediate, s.intermediate)
        _copy_module_exact(f.output, s.output)
    _copy_module_exact(fnet.cls, model.cls)
    return model
