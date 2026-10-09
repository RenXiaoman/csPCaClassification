"""Standalone LMTTM-style 3D classifier adapted from LMTTM-VMI.

The original project targets MedMNIST3D and keeps memory state tied to a
fixed batch size. This implementation keeps the linked-memory read/process/
write idea while making the state per input batch and device agnostic.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class _PatchEmbed3D(nn.Module):
    def __init__(self, in_channels: int, dim: int, patch_size: Sequence[int]) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, dim, kernel_size=tuple(patch_size), stride=tuple(patch_size))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, D, H, W] -> [B, D', tokens, dim]
        x = F.gelu(self.proj(x))
        x = x.flatten(3).permute(0, 2, 3, 1)
        return self.norm(x)


class _LinkedMemoryUnit(nn.Module):
    def __init__(self, dim: int, memory_tokens: int, input_tokens: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.memory_tokens = memory_tokens
        self.read = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.process = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 3,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
        )
        self.write = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.memory_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)
        self.memory_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.input_tokens = input_tokens

    def forward(self, memory: torch.Tensor, current: torch.Tensor, previous: torch.Tensor, following: torch.Tensor):
        context = torch.cat((previous, current, following), dim=1)
        read, _ = self.read(current, context, context, need_weights=False)
        processed = self.process(torch.cat((current, read, context[:, : self.input_tokens]), dim=1))
        output = self.output_norm(processed[:, : self.input_tokens] + current)
        update, _ = self.write(memory, processed, processed, need_weights=False)
        gate = self.memory_gate(torch.cat((memory, update), dim=-1))
        next_memory = self.memory_norm(memory + gate * update)
        return next_memory, output


class LMTTMVMI(nn.Module):
    """Standalone 3D LMTTM classifier for MRI volumes.

    Input shape is ``[B, C, D, H, W]``. The default patching maps the current
    16x256x256 volumes to 8 depth steps with 64 tokens per step.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        dim: int = 128,
        patch_size: Sequence[int] = (2, 64, 64),
        memory_tokens: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % 8:
            raise ValueError("dim must be divisible by 8")
        patch_size = tuple(int(v) for v in patch_size)
        self.patch_embed = _PatchEmbed3D(in_channels, dim, patch_size)
        self.memory_tokens = memory_tokens
        self.memory_init = nn.Parameter(torch.zeros(1, memory_tokens, dim))
        nn.init.normal_(self.memory_init, std=0.02)
        self.position = nn.Parameter(torch.zeros(1, 512, dim))
        nn.init.normal_(self.position, std=0.02)
        # For the default 256x256 input this produces 16 tokens per depth step.
        token_count = (256 // patch_size[1]) * (256 // patch_size[2])
        self.unit = _LinkedMemoryUnit(dim, memory_tokens, token_count, 8, dropout)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        b, steps, token_count, dim = tokens.shape
        if token_count != self.unit.input_tokens:
            raise ValueError(f"expected {self.unit.input_tokens} tokens per step, got {token_count}; adjust patch_size")
        if steps > self.position.shape[1]:
            raise ValueError(f"too many depth steps ({steps}) for positional embedding")
        memory = self.memory_init.expand(b, -1, -1)
        outputs = []
        for i in range(steps):
            current = tokens[:, i] + self.position[:, :token_count]
            previous = tokens[:, i - 1] + self.position[:, :token_count] if i else current
            following = tokens[:, i + 1] + self.position[:, :token_count] if i + 1 < steps else current
            memory, output = self.unit(memory, current, previous, following)
            outputs.append(output)
        summary = torch.cat(outputs, dim=1).mean(dim=1)
        return self.head(summary)
