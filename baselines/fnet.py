import types

import torch
from transformers import FNetConfig, FNetForMaskedLM


def base_fnet_config():
    return FNetConfig(
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
    )


def patch_fnet_fft_fp32(model):
    """Run FNet FFTs in FP32 under CUDA AMP, matching the recorded runs."""
    def fft_forward(self, hidden_states):
        original_dtype = hidden_states.dtype
        if hidden_states.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                result = torch.fft.fftn(hidden_states.float(), dim=(1, 2)).real
        else:
            result = torch.fft.fftn(hidden_states.float(), dim=(1, 2)).real
        return (result.to(dtype=original_dtype),)

    patched = 0
    for layer in model.fnet.encoder.layer:
        layer.fourier.self.forward = types.MethodType(fft_forward, layer.fourier.self)
        patched += 1
    if patched != model.config.num_hidden_layers:
        raise RuntimeError(f"FNet FFT patch count={patched}")
    return model


def build_fnet(seed=2026):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cfg = base_fnet_config()
    cfg.use_tpu_fourier_optimizations = False
    model = FNetForMaskedLM(cfg)
    model.fnet.pooler = None
    return patch_fnet_fft_fp32(model)
