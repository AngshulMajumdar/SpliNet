import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 32000
SEQ_LENGTH = 512
HIDDEN_SIZE = 768
NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 64
FFN_SIZE = 3072
TYPE_VOCAB_SIZE = 4
DROPOUT = 0.1
ATTENTION_DROPOUT = 0.1
LAYER_NORM_EPS = 1e-12
INITIALIZER_RANGE = 0.02
PAD_ID = 3
TRANSFORMER_SEED = 2026


def gelu_new(x):
    return 0.5 * x * (
        1.0
        + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
        )
    )


class TransformerEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE, padding_idx=PAD_ID)
        self.position_embeddings = nn.Embedding(SEQ_LENGTH, HIDDEN_SIZE)
        self.token_type_embeddings = nn.Embedding(TYPE_VOCAB_SIZE, HIDDEN_SIZE)
        self.LayerNorm = nn.LayerNorm(HIDDEN_SIZE, eps=LAYER_NORM_EPS)
        self.dropout = nn.Dropout(DROPOUT)
        self.register_buffer(
            "position_ids",
            torch.arange(SEQ_LENGTH).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids):
        B, L = input_ids.shape
        position_ids = self.position_ids[:, :L]
        token_type_ids = torch.zeros((B, L), dtype=torch.long, device=input_ids.device)
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.dropout(self.LayerNorm(x))


class FastSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(HIDDEN_SIZE, 3 * HIDDEN_SIZE, bias=True)
        self.out_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.LayerNorm = nn.LayerNorm(HIDDEN_SIZE, eps=LAYER_NORM_EPS)

    def forward(self, x):
        B, L, D = x.shape
        qkv = (
            self.qkv(x)
            .view(B, L, 3, NUM_HEADS, HEAD_DIM)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(dim=0)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=ATTENTION_DROPOUT if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        y = self.dropout(self.out_proj(y))
        return self.LayerNorm(x + y)


class TransformerFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_in = nn.Linear(HIDDEN_SIZE, FFN_SIZE)
        self.dense_out = nn.Linear(FFN_SIZE, HIDDEN_SIZE)
        self.dropout = nn.Dropout(DROPOUT)
        self.LayerNorm = nn.LayerNorm(HIDDEN_SIZE, eps=LAYER_NORM_EPS)

    def forward(self, x):
        y = self.dense_out(gelu_new(self.dense_in(x)))
        y = self.dropout(y)
        return self.LayerNorm(x + y)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = FastSelfAttention()
        self.ffn = TransformerFFN()

    def forward(self, x):
        return self.ffn(self.attention(x))


class MLMHead(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.dense = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.LayerNorm = nn.LayerNorm(HIDDEN_SIZE, eps=LAYER_NORM_EPS)
        self.bias = nn.Parameter(torch.zeros(VOCAB_SIZE))
        self.embeddings = embeddings

    def forward(self, x):
        x = self.LayerNorm(gelu_new(self.dense(x)))
        return F.linear(x, self.embeddings.weight, self.bias)


class StandardTransformerMLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = TransformerEmbeddings()
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYERS)])
        self.mlm = MLMHead(self.embeddings.word_embeddings)

    def encode(self, input_ids):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, input_ids, labels=None):
        logits = self.mlm(self.encode(input_ids))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return loss, logits


def _init_transformer_module(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=INITIALIZER_RANGE)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=INITIALIZER_RANGE)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def build_transformer(seed=TRANSFORMER_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = StandardTransformerMLM()
    model.apply(_init_transformer_module)
    return model
