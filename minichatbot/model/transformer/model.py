"""Decoder-only transformer language model.

Composition: token embedding -> N x TransformerBlock -> final norm -> lm_head.
RoPE is applied inside attention; there are no learned position embeddings.
The lm_head shares its weight with the token embedding when
`cfg.tie_embeddings` is true.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from minichatbot.config import ModelConfig
from minichatbot.model import MODEL_REGISTRY
from minichatbot.model.base import LanguageModel, ModelOutput
from minichatbot.model.transformer.attention import KVCache, precompute_rope_cache
from minichatbot.model.transformer.block import TransformerBlock
from minichatbot.model.transformer.norm import make_norm

TransformerState = list[KVCache]


@MODEL_REGISTRY.register("transformer")
class Transformer(LanguageModel):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_layers = cfg.n_layers
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads

        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_dropout = (
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    d_ff=cfg.d_ff,
                    dropout=cfg.dropout,
                    norm_type=cfg.norm_type,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_norm = make_norm(cfg.norm_type, cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_embed.weight

        cos, sin = precompute_rope_cache(cfg.max_seq_len, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> TransformerState:
        return [
            KVCache(
                k=torch.empty(
                    batch_size, self.n_heads, 0, self.head_dim, device=device
                ),
                v=torch.empty(
                    batch_size, self.n_heads, 0, self.head_dim, device=device
                ),
            )
            for _ in range(self.n_layers)
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        state: TransformerState | None = None,
    ) -> ModelOutput:
        B, T = input_ids.shape
        pos_start = state[0].k.size(2) if state is not None else 0
        if pos_start + T > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {pos_start + T} exceeds "
                f"max_seq_len={self.cfg.max_seq_len}"
            )

        x = self.tok_embed(input_ids)
        x = self.embed_dropout(x)

        new_state: TransformerState | None = (
            list(state) if state is not None else None
        )
        for i, block in enumerate(self.blocks):
            cache = state[i] if state is not None else None
            x, new_cache = block(x, self.rope_cos, self.rope_sin, cache)
            if new_state is not None and new_cache is not None:
                new_state[i] = new_cache

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return ModelOutput(logits=logits, state=new_state)

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> Transformer:
        return cls(cfg)
