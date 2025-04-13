#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NeighborRetr model for cross-modal retrieval that addresses the hubness problem.

This module implements the NeighborRetr architecture as described in 
"NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval".

Key components:
1. Centrality Weighting Loss - to emphasize hub centrality
2. Neighbor Adjusting Loss - to balance good and bad hubs
3. Uniform Regularization Loss - to address anti-hub issues
"""
import os
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import CrossModel, Transformer as TransformerClip
from .until_module import (
    LayerNorm, AllGather, AllGather2,
    CentralityWeightingLoss, UniformRegularizationLoss, 
    NeighborAdjustingLoss, KLDivergenceLoss
)
from .cluster import CTM, TCBlock

# Define global allgather functions
allgather = AllGather.apply
allgather2 = AllGather2.apply


class ResidualLinear(nn.Module):
    def __init__(self, dim):
        super(ResidualLinear, self).__init__()
        self.fc_relu = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return x + self.fc_relu(x)


class NeighborRetr(nn.Module):
    def __init__(self, config):
        super(NeighborRetr, self).__init__()

        self.config = config
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        # Initialize components
        self._init_clip_model(backbone)
        self._init_cross_model_config()
        self._init_weighting_networks()
        self._init_video_transformer()
        self._init_loss_functions()
        self._init_memory_bank()
        
        # Apply weight initialization
        self.apply(self._init_weights)
        
        # Load CLIP state dict
        self.clip.load_state_dict(self.clip_state_dict, strict=False)

        # Initialize token clustering and position embeddings
        self._init_token_clustering()
        self._init_frame_position_embeddings()

    def _init_clip_model(self, backbone):
        # Validate backbone exists
        assert backbone in _PT_NAME, f"Backbone {backbone} not supported"
        
        # Load model weights
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist")
            
        try:
            # Try loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            self.clip_state_dict = model.state_dict()
        except RuntimeError:
            # Fall back to regular loading
            self.clip_state_dict = torch.load(model_path, map_location="cpu")

        # Extract model dimensions
        vision_width = self.clip_state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in self.clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = self.clip_state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((self.clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = self.clip_state_dict["text_projection"].shape[1]
        context_length = self.clip_state_dict["positional_embedding"].shape[0]
        vocab_size = self.clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = self.clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in self.clip_state_dict if k.startswith(f"transformer.resblocks")))

        # Initialize CLIP model
        self.clip = CLIP(
            embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )

        # Store model dimensions
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.context_length = context_length

        # Convert to fp16 if CUDA is available
        if torch.cuda.is_available():
            convert_weights(self.clip)

    def _init_cross_model_config(self):
        # Cross-modal transformer configuration
        cross_config = SimpleNamespace(**{
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 2048,
            "max_position_embeddings": self.context_length,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": 512,
            "soft_t": 0.07,
        })

        cross_config.hidden_size = self.transformer_width
        self.cross_config = cross_config

    def _init_weighting_networks(self):
        # Token weighting networks
        self.text_weight_fc = self._create_weighting_network()
        self.video_weight_fc = self._create_weighting_network()
        self.text_weight_fc0 = self._create_weighting_network()
        self.video_weight_fc0 = self._create_weighting_network()
        self.text_weight_fc1 = self._create_weighting_network()
        self.video_weight_fc1 = self._create_weighting_network()
        self.text_weight_intra = self._create_weighting_network()
        self.video_weight_intra = self._create_weighting_network()

    def _create_weighting_network(self):
        return nn.Sequential(
            nn.Linear(self.transformer_width, 2 * self.transformer_width),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.transformer_width, 1)
        )

    def _init_video_transformer(self):
        # Position embeddings for video frames
        self.frame_position_embeddings = nn.Embedding(
            self.cross_config.max_position_embeddings,
            self.cross_config.hidden_size
        )
        
        # Transformer for video frame feature aggregation
        self.transformerClip = TransformerClip(
            width=self.transformer_width,
            layers=self.config.num_hidden_layers,
            heads=self.transformer_heads
        )

    def _init_loss_functions(self):
        self.centrality_weighting_loss = CentralityWeightingLoss()
        self.neighbor_adjusting_loss = NeighborAdjustingLoss()
        self.uniform_regularization_loss = UniformRegularizationLoss()
        self.kl_loss = KLDivergenceLoss()

    def _init_memory_bank(self):
        device = torch.device("cpu")  # Will be moved to correct device later
        
        # Create empty tensors for the memory bank
        self.mb_ind = torch.tensor([], dtype=torch.long, device=device)
        self.mb_feat_t = torch.empty((0, 0, 0), dtype=torch.float, device=device)
        self.mb_feat_v = torch.empty((0, 0, 0), dtype=torch.float, device=device)
        self.mb_mask_t = torch.empty((0, 0), dtype=torch.float, device=device)
        self.mb_mask_v = torch.empty((0, 0), dtype=torch.float, device=device)
        self.mb_batch = 0

    def _init_token_clustering(self):
        # Text token merging layers
        self.text_ctm0 = CTM(sample_ratio=1/6, embed_dim=512, dim_out=512, k=3)
        self.text_block0 = TCBlock(dim=512, num_heads=8)
        self.text_ctm1 = CTM(sample_ratio=1/4, embed_dim=512, dim_out=512, k=3)
        self.text_block1 = TCBlock(dim=512, num_heads=8)

        # Video token merging layers
        self.video_ctm0 = CTM(sample_ratio=1/4, embed_dim=512, dim_out=512, k=3)
        self.video_block0 = TCBlock(dim=512, num_heads=8)
        self.video_ctm1 = CTM(sample_ratio=1/3, embed_dim=512, dim_out=512, k=3)
        self.video_block1 = TCBlock(dim=512, num_heads=8)

    def _init_frame_position_embeddings(self):
        # Initialize frame position embeddings from CLIP positional embeddings
        new_state_dict = OrderedDict()
        contain_frame_position = False
        
        for key in self.clip_state_dict.keys():
            if key.find("frame_position_embeddings") > -1:
                contain_frame_position = True
                break
                
        if contain_frame_position is False:
            for key, val in self.clip_state_dict.items():
                if key == "positional_embedding":
                    new_state_dict["frame_position_embeddings.weight"] = val.clone()
                    continue
                if key.find("transformer.resblocks") == 0:
                    num_layer = int(key.split(".")[2])
                    if num_layer < self.config.num_hidden_layers:
                        new_state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                        continue

        self.load_state_dict(new_state_dict, strict=False)

    def update_memory_bank(self, idx, text_feat, video_feat, text_mask, video_mask):
        # Initialize memory bank if empty
        if self.mb_feat_v.size(0) == 0:
            self.mb_ind = idx.clone()
            self.mb_feat_v = video_feat.clone()
            self.mb_feat_t = text_feat.clone()
            self.mb_mask_t = text_mask.clone()
            self.mb_mask_v = video_mask.clone()
            self.mb_batch = idx.size(0)
            return
        
        # Get current memory bank capacity
        mb_capacity = self.mb_feat_v.size(0)
        
        # Concatenate new samples to memory bank (current batch goes first)
        self.mb_ind = torch.cat((idx, self.mb_ind), dim=0)
        self.mb_feat_v = torch.cat((video_feat, self.mb_feat_v), dim=0)
        self.mb_feat_t = torch.cat((text_feat, self.mb_feat_t), dim=0)
        self.mb_mask_t = torch.cat((text_mask, self.mb_mask_t), dim=0)
        self.mb_mask_v = torch.cat((video_mask, self.mb_mask_v), dim=0)
        
        # Maintain memory bank size (FIFO queue)
        if self.mb_ind.size(0) > mb_capacity:
            self.mb_ind = self.mb_ind[:mb_capacity]
            self.mb_feat_v = self.mb_feat_v[:mb_capacity]
            self.mb_feat_t = self.mb_feat_t[:mb_capacity]
            self.mb_mask_t = self.mb_mask_t[:mb_capacity]
            self.mb_mask_v = self.mb_mask_v[:mb_capacity]

    def forward(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0, logger=None):
        # Reshape inputs
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        
        # Ensure video has proper type and shape
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        # Get text and video features
        text_feat, video_feat = self.get_text_video_feat(text_ids, text_mask, video, video_mask, shaped=True)

        if not self.training:
            # During evaluation, we don't compute losses
            return None

        # Gather features across GPUs for distributed training
        if torch.cuda.is_available():
            idx = allgather(idx, self.config)
            text_feat = allgather(text_feat, self.config)
            video_feat = allgather(video_feat, self.config)
            text_mask = allgather(text_mask, self.config)
            video_mask = allgather(video_mask, self.config)
            torch.distributed.barrier()  # Force sync

        # Get logit scale from CLIP
        logit_scale = self.clip.logit_scale.exp()

        # Extract hyperparameters
        centrality_scale = self.config.centrality_scale
        beta = self.config.beta
        num_neighbors = self.config.num_neighbors
        temperature = self.config.temperature
        
        # Get memory bank features
        mb_feat_t = self.mb_feat_t
        mb_feat_v = self.mb_feat_v
        mb_mask_t = self.mb_mask_t
        mb_mask_v = self.mb_mask_v

        # Calculate losses
        losses = self._compute_losses(
            text_feat, video_feat, 
            text_mask, video_mask,
            mb_feat_t, mb_feat_v,
            mb_mask_t, mb_mask_v,
            centrality_scale, beta, 
            num_neighbors, temperature,
            logit_scale
        )
        
        # Update memory bank queue
        with torch.no_grad():
            self.update_memory_bank(idx, text_feat, video_feat, text_mask, video_mask)

        return losses

    def _compute_losses(self, text_feat, video_feat, text_mask, video_mask,
                        mb_feat_t, mb_feat_v, mb_mask_t, mb_mask_v,
                        centrality_scale, beta, num_neighbors, temperature,
                        logit_scale):
        # Local-level matching (token-wise interaction)
        local_t2v_logits, local_v2t_logits = self.local_level(
            text_feat, video_feat, text_mask, video_mask
        )

        # Global-level matching and uniform regularization loss
        uniform_loss, global_text_feat, global_video_feat, global_t2v_logits, global_v2t_logits = self.compute_uniform_loss(
            text_feat, video_feat, text_mask, video_mask, temperature, beta
        )

        # KL divergence loss between local-level and global-level distributions
        kl_loss = (
            self.kl_loss(global_t2v_logits, local_t2v_logits) + 
            self.kl_loss(global_v2t_logits, local_v2t_logits)
        ) / 2
        
        # Centrality weighting loss
        centrality_loss = self.compute_centrality_loss(
            text_feat, video_feat, 
            global_text_feat, global_video_feat, 
            local_t2v_logits, local_v2t_logits,
            centrality_scale, logit_scale
        )

        # Neighbor adjusting loss
        neighbor_loss = self.compute_neighbor_loss(
            text_feat, video_feat, 
            text_mask, video_mask,
            mb_feat_t, mb_feat_v,
            mb_mask_t, mb_mask_v,
            local_t2v_logits, local_v2t_logits,
            num_neighbors, temperature
        )

        # Combine all losses with their respective weights
        total_loss = (
            centrality_loss + 
            (uniform_loss * self.config.uniform_weight) +
            (neighbor_loss * self.config.neighbor_weight) +
            (kl_loss * self.config.kl_weight)
        )
        
        return total_loss, centrality_loss, uniform_loss, neighbor_loss, kl_loss

    def compute_centrality_loss(self, text_feat, video_feat, 
                               global_text_feat, global_video_feat, 
                               local_t2v_logits, local_v2t_logits,
                               centrality_scale, logit_scale):
        # Calculate local centrality weights
        local_text_weights, local_video_weights = self.compute_centrality_weights(
            text_feat, video_feat, global_text_feat, global_video_feat, centrality_scale
        )
        
        # Apply centrality weighting to entity-level logits
        centrality_loss_t2v = self.centrality_weighting_loss(
            local_t2v_logits * logit_scale, local_text_weights
        )
        centrality_loss_v2t = self.centrality_weighting_loss(
            local_v2t_logits * logit_scale, local_video_weights
        )
        
        # Average bidirectional losses
        return (centrality_loss_t2v + centrality_loss_v2t) / 2

    def compute_neighbor_loss(self, text_feat, video_feat, 
                             text_mask, video_mask,
                             mb_feat_t, mb_feat_v,
                             mb_mask_t, mb_mask_v,
                             local_t2v_logits, local_v2t_logits,
                             num_neighbors, temperature):
        # Calculate memory bank logits
        memory_bank_t2v_logits, _ = self.local_level(text_feat, mb_feat_v, text_mask, mb_mask_v)
        _, memory_bank_v2t_logits = self.local_level(mb_feat_t, video_feat, mb_mask_t, video_mask)

        # Apply neighbor adjusting loss
        neighbor_loss_t2v = self.neighbor_adjusting_loss(
            local_t2v_logits, memory_bank_v2t_logits, num_neighbors, temperature
        )
        neighbor_loss_v2t = self.neighbor_adjusting_loss(
            local_v2t_logits, memory_bank_t2v_logits, num_neighbors, temperature
        )
        
        # Average bidirectional losses
        return (neighbor_loss_t2v + neighbor_loss_v2t) / 2

    def compute_centrality_weights(self, text_feat, video_feat, 
                                  global_text_feat, global_video_feat, 
                                  centrality_scale):
        dim = text_feat.size(2)
        text_feat = text_feat.reshape(-1, dim)
        video_feat = video_feat.reshape(-1, dim)
        global_text_feat = global_text_feat.squeeze(1)
        global_video_feat = global_video_feat.squeeze(1)

        # Normalize features for cosine similarity
        text_feat = F.normalize(text_feat, dim=-1)
        video_feat = F.normalize(video_feat, dim=-1)
        global_text_feat = F.normalize(global_text_feat, dim=-1)
        global_video_feat = F.normalize(global_video_feat, dim=-1)

        # Calculate similarity between global and local features
        text_global_local_sim = torch.matmul(global_text_feat, text_feat.permute(1, 0))
        video_global_local_sim = torch.matmul(global_video_feat, video_feat.permute(1, 0))

        # Calculate average similarity (centrality score)
        text_centrality = torch.mean(text_global_local_sim, dim=-1)
        video_centrality = torch.mean(video_global_local_sim, dim=-1)

        # Apply exponential scaling to get final weights
        text_weights = torch.exp(text_centrality * centrality_scale)
        video_weights = torch.exp(video_centrality * centrality_scale)

        return text_weights, video_weights
    
    def compute_uniform_loss(self, text_feat, video_feat, text_mask, video_mask, temperature, beta):
        # Get global features through hierarchical token merging
        global_text_feat, global_video_feat = self.merge_global_features(text_feat, video_feat, text_mask, video_mask)

        # Calculate similarity logits at global level
        global_t2v_logits, global_v2t_logits = self.global_level(global_text_feat, global_video_feat)

        # Calculate uniform regularization loss
        uniform_loss_t2v = self.uniform_regularization_loss(global_t2v_logits, temperature, beta)
        uniform_loss_v2t = self.uniform_regularization_loss(global_v2t_logits, temperature, beta)
        uniform_loss = (uniform_loss_t2v + uniform_loss_v2t) / 2

        return uniform_loss, global_text_feat, global_video_feat, global_t2v_logits, global_v2t_logits

    def merge_global_features(self, text_feat, video_feat, text_mask, video_mask):
        # Prepare token dictionaries for text
        t_idx_token = torch.arange(text_feat.size(1))[None, :].repeat(text_feat.size(0), 1)
        t_agg_weight = text_feat.new_ones(text_feat.size(0), text_feat.size(1), 1)
        t_token_dict = {
            'x': text_feat,
            'token_num': text_feat.size(1),
            'idx_token': t_idx_token,
            'agg_weight': t_agg_weight,
            'mask': text_mask.detach()
        }

        # Prepare token dictionaries for video
        v_idx_token = torch.arange(video_feat.size(1))[None, :].repeat(video_feat.size(0), 1)
        v_agg_weight = video_feat.new_ones(video_feat.size(0), video_feat.size(1), 1)
        v_token_dict = {
            'x': video_feat,
            'token_num': video_feat.size(1),
            'idx_token': v_idx_token,
            'agg_weight': v_agg_weight,
            'mask': video_mask.detach()
        }

        # First level of token merging
        t_token_dict = self.text_block0(self.text_ctm0(t_token_dict))
        v_token_dict = self.video_block0(self.video_ctm0(v_token_dict))
        text_feat = t_token_dict["x"]
        video_feat = v_token_dict["x"]

        # Second level of token merging
        t_token_dict = self.text_block1(self.text_ctm1(t_token_dict))
        v_token_dict = self.video_block1(self.video_ctm1(v_token_dict))
        text_feat = t_token_dict["x"]
        video_feat = v_token_dict["x"]

        return text_feat, video_feat

    def local_level(self, text_feat, video_feat, text_mask, video_mask):
        # Calculate attention weights for text tokens
        text_weight = self.text_weight_fc(text_feat).squeeze(2)  # [B, N_t]
        text_weight.masked_fill_((1 - text_mask).to(torch.bool), float(-9e15))
        text_weight = torch.softmax(text_weight, dim=-1)  # [B, N_t]

        # Calculate attention weights for video tokens
        video_weight = self.video_weight_fc(video_feat).squeeze(2)  # [B, N_v]
        video_weight.masked_fill_((1 - video_mask).to(torch.bool), float(-9e15))
        video_weight = torch.softmax(video_weight, dim=-1)  # [B, N_v]

        # Normalize features
        text_feat = F.normalize(text_feat, dim=-1)
        video_feat = F.normalize(video_feat, dim=-1)

        # Calculate similarity between all text and video tokens
        retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat, video_feat])
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask])
        retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])

        # Calculate text-to-video similarity
        t2v_logits, _ = retrieve_logits.max(dim=-1)  # [B, B, N_t]
        t2v_similarity = torch.einsum('abt,at->ab', [t2v_logits, text_weight])  # [B, B]

        # Calculate video-to-text similarity
        v2t_logits, _ = retrieve_logits.max(dim=-2)  # [B, B, N_v]
        v2t_similarity = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])  # [B, B]

        # Average bidirectional similarities
        similarity = (t2v_similarity + v2t_similarity) / 2.0

        return similarity, similarity.T

    def global_level(self, text_feat, video_feat):
        # Calculate attention weights for text tokens
        text_weight = self.text_weight_fc1(text_feat).squeeze(2)  # [B, 1]
        text_weight = torch.softmax(text_weight, dim=-1)  # [B, 1]

        # Calculate attention weights for video tokens
        video_weight = self.video_weight_fc1(video_feat).squeeze(2)  # [B, 1]
        video_weight = torch.softmax(video_weight, dim=-1)  # [B, 1]

        # Calculate similarity between all text and video global features
        retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat, video_feat])  # [B, B, 1, 1]

        # Calculate text-to-video similarity
        t2v_logits, _ = retrieve_logits.max(dim=-1)  # [B, B, 1]
        t2v_similarity = torch.einsum('abt,at->ab', [t2v_logits, text_weight])  # [B, B]

        # Calculate video-to-text similarity
        v2t_logits, _ = retrieve_logits.max(dim=-2)  # [B, B, 1]
        v2t_similarity = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])  # [B, B]

        # Average bidirectional similarities
        similarity = (t2v_similarity + v2t_similarity) / 2.0

        return similarity, similarity.T

    def get_text_feat(self, text_ids, text_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])

        bs_pair = text_ids.size(0)
        _, text_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask)
        text_feat = text_feat.float()
        text_feat = text_feat.view(bs_pair, -1, text_feat.size(-1))
        return text_feat

    def get_video_feat(self, video, video_mask, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.size()) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        bs_pair, n_v = video_mask.size()
        video_feat = self.clip.encode_image(video, return_hidden=True)[0].float()
        video_feat = video_feat.float().view(bs_pair, -1, video_feat.size(-1))
        video_feat = self.aggregate_video_features(video_feat, video_mask)
        return video_feat

    def get_text_video_feat(self, text_ids, text_mask, video, video_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.shape) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        text_feat = self.get_text_feat(text_ids, text_mask, shaped=True)
        video_feat = self.get_video_feat(video, video_mask, shaped=True)

        return text_feat, video_feat

    def get_video_avg_feat(self, video_feat, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_feat = video_feat * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_feat = torch.sum(video_feat, dim=1) / video_mask_un_sum
        return video_feat

    def get_text_sep_feat(self, text_feat, text_mask):
        text_feat = text_feat.contiguous()
        text_feat = text_feat[torch.arange(text_feat.shape[0]), torch.sum(text_mask, dim=-1) - 1, :]
        text_feat = text_feat.unsqueeze(1).contiguous()
        return text_feat

    def aggregate_video_features(self, video_feat, video_mask):
        video_feat = video_feat.contiguous()

        # Add position embeddings
        video_feat_original = video_feat
        seq_length = video_feat.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=video_feat.device)
        position_ids = position_ids.unsqueeze(0).expand(video_feat.size(0), -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        video_feat = video_feat + frame_position_embeddings
        
        # Create attention mask for transformer
        extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
        extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
        
        # Apply transformer encoder
        video_feat = video_feat.permute(1, 0, 2)  # NLD -> LND
        video_feat = self.transformerClip(video_feat, extended_video_mask)
        video_feat = video_feat.permute(1, 0, 2)  # LND -> NLD
        
        # Add residual connection
        video_feat = video_feat + video_feat_original
        return video_feat

    def get_similarity_logits(self, text_feat, video_feat, text_mask, video_mask, shaped=False):
        if shaped is False:
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        local_similarity, _ = self.local_level(text_feat, video_feat, text_mask, video_mask)

        return local_similarity, local_similarity.T

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def _init_weights(self, module):
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