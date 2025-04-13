#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-modal transformer model implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import json
import torch
from torch import nn
from timm.models.layers import drop_path
from collections import OrderedDict

from .until_module import LayerNorm, ACT2FN
from .until_config import PretrainedConfig

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'cross_config.json'
WEIGHTS_NAME = 'cross_pytorch_model.bin'


class QuickGELU(nn.Module):
    """
    Fast approximation of GELU activation function.
    """
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks.
    
    Args:
        drop_prob: Probability of dropping the path
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f'p={self.drop_prob}'


class ResidualAttentionBlock(nn.Module):
    """
    Transformer block with residual attention and MLP.
    
    Args:
        d_model: Model dimension
        n_head: Number of attention heads
        drop_path: Stochastic depth rate
    """
    def __init__(self, d_model: int, n_head: int, drop_path=0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor):
        """
        Multi-head attention with mask.
        
        Args:
            x: Input tensor
            attn_mask: Attention mask
            
        Returns:
            Attention output
        """
        attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, para_tuple: tuple):
        """
        Forward pass through the residual block.
        
        Args:
            para_tuple: Tuple of (x, attn_mask)
            
        Returns:
            Updated (x, attn_mask) tuple
        """
        x, attn_mask = para_tuple
        if self.training:
            x = x + self.drop_path(self.attention(self.ln_1(x), attn_mask))
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        else:
            x = x + self.attention(self.ln_1(x), attn_mask)
            x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)


class Transformer(nn.Module):
    """
    Transformer encoder for cross-modal processing.
    
    Args:
        width: Model dimension
        layers: Number of transformer layers
        heads: Number of attention heads
    """
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        """
        Forward pass through transformer.
        
        Args:
            x: Input tensor [L, N, D]
            attn_mask: Attention mask
            
        Returns:
            Transformer output
        """
        return self.resblocks((x, attn_mask))[0]


class CrossEmbeddings(nn.Module):
    """
    Embeddings module for cross-modal transformer.
    
    Args:
        config: Model configuration
    """
    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, concat_embeddings, concat_type=None):
        """
        Create embeddings with positional information.
        
        Args:
            concat_embeddings: Input embeddings
            concat_type: Type embeddings (unused)
            
        Returns:
            Combined embeddings
        """
        batch_size, seq_length = concat_embeddings.size(0), concat_embeddings.size(1)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=concat_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = concat_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class CrossPooler(nn.Module):
    """
    Pooler for cross-modal transformer outputs.
    
    Args:
        config: Model configuration
    """
    def __init__(self, config):
        super().__init__()
        self.ln_pool = LayerNorm(config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = QuickGELU()

    def forward(self, hidden_states, hidden_mask):
        """
        Pool transformer outputs to a single vector.
        
        Args:
            hidden_states: Transformer output states
            hidden_mask: Attention mask
            
        Returns:
            Pooled output vector
        """
        hidden_states = self.ln_pool(hidden_states)
        pooled_output = hidden_states[:, 0]
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class CrossModel(nn.Module):
    """
    Cross-modal transformer model.
    
    Args:
        config: Model configuration
    """
    def initialize_parameters(self):
        """
        Initialize model parameters.
        """
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = CrossEmbeddings(config)

        transformer_width = config.hidden_size
        transformer_layers = config.num_hidden_layers
        transformer_heads = config.num_attention_heads
        self.transformer = Transformer(
            width=transformer_width, 
            layers=transformer_layers, 
            heads=transformer_heads
        )
        self.pooler = CrossPooler(config)
        self.apply(self.init_weights)

    def build_attention_mask(self, attention_mask):
        """
        Build attention mask for transformer.
        
        Args:
            attention_mask: Input attention mask
            
        Returns:
            Extended attention mask
        """
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0
        extended_attention_mask = extended_attention_mask.expand(-1, attention_mask.size(1), -1)
        return extended_attention_mask

    def forward(self, concat_input, concat_type=None, attention_mask=None, output_all_encoded_layers=True):
        """
        Forward pass through cross-modal transformer.
        
        Args:
            concat_input: Input tensor
            concat_type: Type tensor (optional)
            attention_mask: Attention mask (optional)
            output_all_encoded_layers: Whether to output all layers (unused)
            
        Returns:
            Tuple of (sequence_output, pooled_output)
        """
        if attention_mask is None:
            attention_mask = torch.ones(concat_input.size(0), concat_input.size(1))
        if concat_type is None:
            concat_type = torch.zeros_like(attention_mask)

        extended_attention_mask = self.build_attention_mask(attention_mask)

        embedding_output = self.embeddings(concat_input, concat_type)
        embedding_output = embedding_output.permute(1, 0, 2)  # NLD -> LND
        embedding_output = self.transformer(embedding_output, extended_attention_mask)
        embedding_output = embedding_output.permute(1, 0, 2)  # LND -> NLD

        pooled_output = self.pooler(embedding_output, hidden_mask=attention_mask)

        return embedding_output, pooled_output

    @property
    def dtype(self):
        """
        Get model data type.
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        """
        Initialize module weights.
        
        Args:
            module: Module to initialize
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class CrossConfig(PretrainedConfig):
    """
    Configuration class for CrossModel.
    
    Args:
        vocab_size_or_config_json_file: Vocabulary size or path to config
        hidden_size: Size of hidden layers
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: Size of feed-forward layers
        hidden_act: Activation function
        hidden_dropout_prob: Dropout probability for hidden layers
        attention_probs_dropout_prob: Dropout probability for attention
        max_position_embeddings: Maximum sequence length
        type_vocab_size: Vocabulary size for token types
        initializer_range: Range for weight initialization
    """
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    config_name = CONFIG_NAME
    weights_name = WEIGHTS_NAME
    
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """
        Initialize CrossConfig with parameters or from a config file.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int) "
                             "or the path to a pretrained model config file (str)")