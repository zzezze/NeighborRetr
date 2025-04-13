#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimizer setup for NeighborRetr.

This module provides functions to set up the optimizer and scheduler for training.
"""
import torch
from NeighborRetr.models.optimization import BertAdam


def prep_optimizer(args, model, num_train_optimization_steps, local_rank):
    """
    Prepare optimizer and scheduler for training.
    
    This function sets up the optimizer with appropriate parameter groups and
    learning rates, and wraps the model for distributed training if needed.
    
    Args:
        args: Command line arguments
        model: Model to optimize
        num_train_optimization_steps: Total number of optimization steps
        local_rank: Local rank for distributed training
        
    Returns:
        tuple: (optimizer, scheduler, model)
    """
    # Get reference to model without DistributedDataParallel wrapper
    if hasattr(model, 'module'):
        model = model.module
        
    # Extract optimizer parameters from args
    lr = args.lr  # Base learning rate (0.0001)
    coef_lr = args.coef_lr  # CLIP-specific learning rate coefficient (0.001)
    weight_decay = args.weight_decay  # Weight decay (0.2)
    warmup_proportion = args.warmup_proportion  # Proportion of training for warmup
    
    # Group parameters based on module name and decay settings
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # Separate parameters into decay/no-decay and CLIP/non-CLIP groups
    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    # Create parameter groups with appropriate hyperparameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    # No explicit scheduler, as it's built into BertAdam
    scheduler = None
    
    # Create optimizer with warmup and cosine schedule
    optimizer = BertAdam(
        optimizer_grouped_parameters, 
        lr=args.lr, 
        warmup=warmup_proportion,
        schedule='warmup_cosine', 
        b1=0.9, 
        b2=0.98, 
        e=1e-6,
        t_total=num_train_optimization_steps, 
        weight_decay=weight_decay,
        max_grad_norm=1.0
    )

    # Set up distributed data parallel model if CUDA is available
    if torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=True
        )
    
    return optimizer, scheduler, model
