#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch module implementations for NeighborRetr model components.
This file includes various loss functions and utility classes for the model.
"""

import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .until_config import PretrainedConfig

logger = logging.getLogger(__name__)


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


##################################
###### LOSS FUNCTIONS ############
##################################


class NeighborAdjustingLoss(nn.Module):
    """
    Implementation of Neighbor Adjusting Loss (L_Nbi) as described in Section 4.3.
    This loss function balances neighborhood relations by differentiating various kinds 
    of affinity in the neighborhood, promoting good hubs and penalizing bad ones.
    """
    def __init__(self, config=None):
        super(NeighborAdjustingLoss, self).__init__()

    def normalize_similarity(self, similarity, mask):
        """
        Apply min-max normalization to similarity matrix.
        
        Args:
            similarity: Similarity matrix to normalize
            mask: Binary mask indicating valid positions
            
        Returns:
            Normalized similarity matrix with values in [0, 1]
        """
        # Mask for min calculation (replace valid positions with large positive value)
        masked_sim_min = torch.where(mask == 0.0, similarity, torch.tensor(9e15, device=similarity.device, dtype=similarity.dtype))
        min_vals = torch.min(masked_sim_min, dim=-1, keepdim=True)[0]
        
        # Mask for max calculation (replace valid positions with large negative value)
        masked_sim_max = torch.where(mask == 0.0, similarity, torch.tensor(-9e15, device=similarity.device, dtype=similarity.dtype))
        max_vals = torch.max(masked_sim_max, dim=-1, keepdim=True)[0]
        
        # Apply min-max normalization
        normalized_similarity = (similarity - min_vals) / (max_vals - min_vals)
        return normalized_similarity

    def create_neighbor_mask(self, similarity_matrix, num_neighbors):
        """
        Create masks for identifying nearest neighbors for each sample.
        
        Args:
            similarity_matrix: Matrix of similarity scores
            num_neighbors: Number of neighbors to consider for each sample
            
        Returns:
            neighbor_mask: Binary mask indicating top-k neighbors
            extended_mask: Binary mask including diagonal and top-k neighbors
        """
        batch_size = similarity_matrix.size(0)
        dtype = similarity_matrix.dtype
        device = similarity_matrix.device
        
        # Create identity matrix to mask out self-similarity
        identity_mask = torch.eye(batch_size, device=device)
        
        # Exclude self-similarity for neighbor selection
        similarity_without_self = torch.where(
            identity_mask == 0.0, 
            similarity_matrix, 
            torch.tensor(-9e15, device=device, dtype=dtype)
        )
        
        # Initialize empty mask for neighbors
        neighbor_mask = torch.zeros((batch_size, batch_size), device=device)
        
        # Find top-k neighbors for each sample
        _, indices = torch.sort(similarity_without_self, dim=-1, descending=True)
        top_k_indices = indices[:, :num_neighbors].flatten()
        sample_indices = torch.arange(0, batch_size, device=device).unsqueeze(1).expand(-1, num_neighbors).flatten()
        
        # Mark top-k neighbors in the mask
        neighbor_mask[sample_indices, top_k_indices] = 1.0
        
        # Extended mask includes both diagonal and neighbors
        extended_mask = identity_mask.clone()
        extended_mask[sample_indices, top_k_indices] = 1.0
        
        return neighbor_mask, extended_mask

    def compute_positive_weights(self, similarity_matrix, mask, temperature):
        """
        Compute positive weights for neighbors based on similarity.
        
        Args:
            similarity_matrix: Matrix of similarity scores
            mask: Binary mask indicating valid positions
            temperature: Temperature parameter for softmax
            
        Returns:
            Positive weights for each neighbor
        """
        dtype = similarity_matrix.dtype
        device = similarity_matrix.device
        
        # Apply softmax with temperature scaling
        pos_weights = torch.softmax(similarity_matrix * temperature, dim=-1)
        
        # Apply mask to keep only valid positions
        pos_weights = torch.where(
            mask == 1.0, 
            pos_weights, 
            torch.tensor(0.0, device=device, dtype=dtype)
        )
        
        # Set diagonal to 1.0 (ground truth)
        pos_weights.fill_diagonal_(1.0)
        
        return pos_weights

    def forward(self, similarity_matrix, memory_bank_matrix, num_neighbors, temperature):
        """
        Compute Neighbor Adjusting Loss.
        
        Args:
            similarity_matrix: Cross-modal similarity matrix
            memory_bank_matrix: Memory bank similarity matrix for centrality estimation
            num_neighbors: Number of neighbors to consider
            temperature: Temperature parameter for softmax
            
        Returns:
            Neighbor Adjusting Loss value
        """
        dtype = similarity_matrix.dtype
        device = similarity_matrix.device
        
        # Get neighbor masks
        neighbor_mask, extended_mask = self.create_neighbor_mask(similarity_matrix, num_neighbors)
        
        # Compute average centrality score from memory bank
        memory_centrality = memory_bank_matrix.sum(dim=-1) / memory_bank_matrix.size(-1)
        memory_centrality_expanded = memory_centrality.unsqueeze(0).repeat(similarity_matrix.size(0), 1)
        
        # Normalize similarity and memory centrality
        normalized_similarity = self.normalize_similarity(similarity_matrix, extended_mask)
        normalized_centrality = self.normalize_similarity(memory_centrality_expanded, extended_mask)
        
        # Compute de-centrality similarity (Eq. 5 in paper)
        adjusted_similarity = torch.where(
            neighbor_mask == 1.0,
            normalized_similarity - normalized_centrality,
            torch.tensor(-9e15, device=device, dtype=dtype)
        )
        
        # Compute positive weights (Eq. 8 in paper)
        positive_weights = self.compute_positive_weights(adjusted_similarity, neighbor_mask, temperature)
        
        # Apply neighbor mask to similarity matrix (Eq. 7 in paper)
        masked_similarity = torch.where(
            extended_mask == 1.0,
            similarity_matrix,
            torch.tensor(-9e15, device=device, dtype=dtype)
        )
        
        # Compute log probabilities weighted by positive weights (Eq. 6 in paper)
        log_probabilities = F.log_softmax(masked_similarity, dim=-1) * positive_weights
        log_probabilities = -torch.sum(log_probabilities, dim=-1) / torch.sum(positive_weights, dim=-1)
        
        # Return mean loss
        loss = log_probabilities.mean()
        return loss


class UniformRegularizationLoss(nn.Module):
    """
    Implementation of Uniform Regularization Loss (L_Opt) as described in Section 4.4.
    This loss enforces equal retrieval probabilities across all samples, addressing the
    anti-hub issue by ensuring anti-hubs have comparable retrieval probabilities.
    """
    def __init__(self, config=None):
        super(UniformRegularizationLoss, self).__init__()

    def sinkhorn_algorithm(self, scores, beta=0.3, num_iterations=50):
        """
        Perform Sinkhorn algorithm for differentiable optimal transport in log-space.
        
        Args:
            scores: Similarity matrix
            beta: Interpolation parameter between OT and identity solutions
            num_iterations: Number of Sinkhorn iterations
            
        Returns:
            Transport plan Q that satisfies marginal constraints
        """
        with torch.no_grad():
            m, n = scores.shape
            one = scores.new_tensor(1)
            ms, ns = (m * one).to(scores), (n * one).to(scores)
            
            # Initialize dual variables
            norm = -(ms + ns).log()
            log_mu = norm.expand(m)
            log_nu = norm.expand(n)
            
            u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
            
            # Sinkhorn iterations
            for _ in range(num_iterations):
                u = log_mu - torch.logsumexp(scores + v.unsqueeze(0), dim=1)
                v = log_nu - torch.logsumexp(scores + u.unsqueeze(1), dim=0)
            
            # Compute log transport plan
            Z = scores + u.unsqueeze(1) + v.unsqueeze(0)
            Z = Z - norm  # Normalize by M+N
        
        # Compute transport plan
        Q = Z.exp()
        
        # Interpolate with identity solution
        identity_matrix = torch.zeros(scores.size()).to(scores.device)
        identity_matrix.fill_diagonal_(1)
        
        # Final transport plan (Eq. 10 in paper)
        targets = beta * Q + (1 - beta) * identity_matrix
        
        return targets

    def forward(self, similarity_matrix, logit_scale, beta=0.3, num_iterations=50):
        """
        Compute Uniform Regularization Loss using optimal transport.
        
        Args:
            similarity_matrix: Cross-modal similarity matrix
            logit_scale: Scaling factor for similarity logits
            beta: Interpolation parameter between OT and identity solutions
            num_iterations: Number of Sinkhorn iterations
            
        Returns:
            Uniform Regularization Loss value
        """
        # Compute optimal transport plan (Eq. 10 in paper)
        transport_plan = self.sinkhorn_algorithm(similarity_matrix, beta, num_iterations)
        
        # Compute log probabilities (Eq. 12 in paper)
        log_probabilities = F.log_softmax(similarity_matrix * logit_scale, dim=-1) * transport_plan
        
        # Compute loss (Eq. 11 in paper)
        log_probabilities = -torch.sum(log_probabilities, dim=-1)
        loss = log_probabilities.mean()
        
        return loss


class CentralityWeightingLoss(nn.Module):
    """
    Implementation of Centrality Weighting Loss (L_Wti) as described in Section 4.2.
    This loss emphasizes the learning of hubs by incorporating centrality weights
    into the contrastive loss function.
    """
    def __init__(self, config=None):
        super(CentralityWeightingLoss, self).__init__()

    def forward(self, similarity_matrix, centrality_weights):
        """
        Compute Centrality Weighting Loss.
        
        Args:
            similarity_matrix: Cross-modal similarity matrix
            centrality_weights: Weights based on centrality scores (Eq. 3 in paper)
            
        Returns:
            Centrality Weighting Loss value
        """
        # Compute log probabilities (softmax over rows)
        log_probabilities = F.log_softmax(similarity_matrix, dim=-1)
        
        # Extract diagonal elements (matching pairs)
        diagonal_log_probs = torch.diag(log_probabilities)
        
        # Apply centrality weights (Eq. 4 in paper)
        weighted_log_probs = diagonal_log_probs * centrality_weights
        
        # Compute negative log likelihood
        nce_loss = -weighted_log_probs
        
        # Return mean loss
        loss = nce_loss.mean()
        return loss


class KLDivergenceLoss(nn.Module):
    """
    Implementation of KL Divergence Loss (L_KL) as described in Section 4.5.
    This loss ensures consistency between high-level and low-level representations.
    """
    def __init__(self, config=None):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, global_similarity, local_similarity):
        """
        Compute KL Divergence Loss between global and local similarity distributions.
        
        Args:
            global_similarity: High-level (global) similarity matrix
            local_similarity: Low-level (local) similarity matrix
            
        Returns:
            KL Divergence Loss value
        """
        # Compute log probabilities for global similarity
        global_log_probs = F.log_softmax(global_similarity, dim=-1)
        
        # Compute probabilities for local similarity
        local_probs = F.softmax(local_similarity, dim=-1)
        
        # Compute KL divergence (Eq. 15 in paper)
        kl_loss = F.kl_div(global_log_probs, local_probs, reduction='mean')
        
        return kl_loss


##################################
###### UTILITY CLASSES ###########
##################################


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        if args.world_size == 1:
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return tensor
        else:
            output = [torch.empty_like(tensor) for _ in range(args.world_size)]
            torch.distributed.all_gather(output, tensor)
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
        )


class AllGather2(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor with improved gradient handling."""

    # https://github.com/PyTorchLightning/lightning-bolts/blob/8d3fbf7782e3d3937ab8a1775a7092d7567f2933/pl_bolts/models/self_supervised/simclr/simclr_module.py#L20
    @staticmethod
    def forward(ctx, tensor, args):
        if args.world_size == 1:
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return tensor
        else:
            output = [torch.empty_like(tensor) for _ in range(args.world_size)]
            torch.distributed.all_gather(output, tensor)
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)
        return (grad_input[ctx.rank * ctx.batch_size:(ctx.rank + 1) * ctx.batch_size], None)


class PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def resize_token_embeddings(self, new_num_tokens=None):
        raise NotImplementedError

    @classmethod
    def init_preweight(cls, model, state_dict, prefix=None, task_config=None):
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='')

        if prefix is None and (task_config is None or task_config.local_rank == 0):
            logger.info("-" * 20)
            if len(missing_keys) > 0:
                logger.info("Weights of {} not initialized from pretrained model: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
            if len(error_msgs) > 0:
                logger.error("Weights from pretrained model cause errors in {}: {}"
                             .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))

        return model

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = find_tensor_attributes(self)
            if gen:
                first_tuple = next(iter(gen))
                return first_tuple[1].dtype
            
            return torch.float32