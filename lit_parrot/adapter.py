"""Implementation of the paper:

LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention
https://arxiv.org/abs/2303.16199

Port for Lit-Parrot
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self

from lit_parrot.config import Config as BaseConfig
from lit_parrot.model import MLP, Parrot as BaseModel, build_rope_cache, apply_rope


@dataclass
class Config(BaseConfig):
    adapter_prompt_length: int = 10
    adapter_start_layer: int = 2


class CausalSelfAttention(nn.Module):
    """A modification of `lit_parrot.model.CausalSelfAttention` that adds the attention
    over the adaption prompt."""

    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=True)

        if block_idx >= config.adapter_start_layer:
            # adapter embedding layer
            self.adapter_wte = nn.Embedding(config.adapter_prompt_length, config.n_embd)
            # gate for adaption
            self.gating_factor = torch.nn.Parameter(torch.zeros(1))

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.rotary_percentage = config.rotary_percentage
        self.rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.block_idx = block_idx
        self.adapter_prompt_length = config.adapter_prompt_length
        self.adapter_start_layer = config.adapter_start_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)
        head_size = C // self.n_head
        qkv = qkv.view(B, T, self.n_head, 3 * head_size).transpose(1, 2)
        q, k, v = qkv.split(head_size, dim=-1)  # (B, nh, T, hs)

        n_elem = int(self.rotary_percentage * head_size)
        if self.rope_cache is None:
            self.rope_cache = build_rope_cache(self.block_size, n_elem, x.dtype, x.device)
        cos, sin = self.rope_cache
        cos, sin = cos[:T], sin[:T]

        q_roped = apply_rope(q[..., :n_elem], cos, sin)
        k_roped = apply_rope(k[..., :n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., n_elem:]), dim=-1)
        k = torch.cat((k_roped, k[..., n_elem:]), dim=-1)

        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=1.0 / math.sqrt(head_size)
        )

        if self.block_idx >= self.adapter_start_layer:
            prefix = self.adapter_wte.weight.reshape(1, self.adapter_prompt_length, self.n_embd)

            aT = prefix.size(1)
            _, ak, av = self.attn(prefix).split(self.n_embd, dim=2)  # mayby dim=2
            ak = ak.view(1, aT, self.n_head, head_size).repeat(B, 1, 1, 1).transpose(1, 2)
            av = av.view(1, aT, self.n_head, head_size).repeat(B, 1, 1, 1).transpose(1, 2)

            amask = torch.ones(q.shape[-2], ak.shape[-2], dtype=torch.bool, device=x.device)
            ay = F.scaled_dot_product_attention(q, ak, av, attn_mask=amask, dropout_p=0.0, is_causal=False)
            y = y + self.gating_factor * ay

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y


class Block(nn.Module):
    """The implementation is identical to `lit_parrot.model.Block` with the exception that
    we replace the attention layer where adaption is implemented."""

    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, block_idx)
        self.norm_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

        self.parallel_residual = config.parallel_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.parallel_residual:
            x = x + self.attn(self.norm_1(x)) + self.mlp(self.norm_2(x))
        else:
            x = x + self.attn(self.norm_1(x))
            x = x + self.mlp(self.norm_2(x))
        return x


class Parrot(BaseModel):
    """The implementation is identical to `lit_parrot.model.Parrot` with the exception that
    the `Block` saves the layer index and passes it down to the attention layer."""

    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config, i) for i in range(config.n_layer)),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(Config.from_name(name))


def mark_only_adapter_as_trainable(model: Parrot) -> None:
    """Sets `requires_grad=False` for all non-adapter weights."""
    for name, param in model.named_parameters():
        param.requires_grad = "adapter_wte" in name or "gating_factor" in name


def adapter_state_from_state_dict(state_dict: dict) -> dict:
    """Returns the model state dict with only the adapter weights for saving."""
    return {name: param for name, param in state_dict.items() if "adapter_wte" in name or "gating_factor" in name}
