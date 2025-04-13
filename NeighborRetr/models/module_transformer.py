#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer module implementation with cross-attention support.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
from torch import nn
from collections import OrderedDict
from timm.models.layers import drop_path
from .until_module import LayerNorm as BaseLayerNorm, ACT2FN

logger = logging.getLogger(__name__)

# Constants for model loading
PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'cross_config.json'
WEIGHTS_NAME = 'cross_pytorch_model.bin'


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization with automatic type conversion.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor with original dtype
        """
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """
    Approximation of GELU activation function.
    Faster than nn.GELU and close enough in practice.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Quick GELU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Activated tensor
        """
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    Transformer block with self-attention and MLP.
    Includes residual connections and layer normalization.
    """
    
    def __init__(self, d_model: int, n_head: int, attn_mask=None):
        """
        Initialize a ResidualAttentionBlock.
        
        Args:
            d_model: Hidden dimension size
            n_head: Number of attention heads
            attn_mask: Optional attention mask
        """
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head
    
    def attention(self, x: torch.Tensor, attn_mask_: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention with mask.
        
        Args:
            x: Input tensor
            attn_mask_: Attention mask
            
        Returns:
            Self-attention output
        """
        if attn_mask_ is not None:
            attn_mask_ = attn_mask_.repeat_interleave(self.n_head, dim=0)
            attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device)
        
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]
    
    def forward(self, para_tuple: tuple) -> tuple:
        """
        Forward pass for ResidualAttentionBlock.
        
        Args:
            para_tuple: Tuple containing (x, attn_mask)
            
        Returns:
            Tuple containing (output, attn_mask)
        """
        x, attn_mask = para_tuple
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)


class Transformer(nn.Module):
    """
    Transformer model with self-attention blocks.
    """
    
    def __init__(self, width: int, layers: int, heads: int, attn_mask=None):
        """
        Initialize a Transformer.
        
        Args:
            width: Hidden dimension size
            layers: Number of transformer layers
            heads: Number of attention heads
            attn_mask: Optional attention mask
        """
        super().__init__()
        
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Transformer.
        
        Args:
            x: Input tensor
            attn_mask: Attention mask
            
        Returns:
            Transformer output
        """
        return self.resblocks((x, attn_mask))[0]