from __future__ import annotations

from torch import nn

from model import Gemma4AudioConfig


class Gemma4AudioModel(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.config = config

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Audio support is not implemented yet in this repo")
