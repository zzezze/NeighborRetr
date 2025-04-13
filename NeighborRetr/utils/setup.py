#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup utilities for NeighborRetr.

This module provides functions to set up the environment for training,
including random seed initialization, logger setup, and distributed training configuration.
"""
import os
import random
import numpy as np
import torch


def set_seed_logger(args):
    """
    Set random seeds for reproducibility and configure logger.
    
    Args:
        args: Command line arguments
        
    Returns:
        args: Updated command line arguments
    """
    # Set random seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Log effective parameters
    args.logger.info("=" * 80)
    args.logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        args.logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def setup_distributed_environment(args):
    """
    Configure distributed training environment.
    
    Args:
        args: Command line arguments
    
    Returns:
        args: Updated command line arguments
    """
    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    else:
        args.device = torch.device("cpu")
        args.world_size = 1
        
    # Create synchronization point for distributed training
    if torch.cuda.is_available():
        torch.distributed.barrier()
        
    args.logger.info(f"Local rank: {args.local_rank}, world size: {args.world_size}")
    
    return args


def reduce_loss(loss, args):
    """
    Reduce loss across all distributed processes.
    
    Args:
        loss: Loss tensor to reduce
        args: Command line arguments with world_size
        
    Returns:
        torch.Tensor: Reduced loss
    """
    world_size = args.world_size
    if world_size < 2:
        return loss
        
    with torch.no_grad():
        torch.distributed.reduce(loss, dst=0)
        if torch.distributed.get_rank() == 0:
            # Only main process gets accumulated, so only divide by
            # world_size in this case
            loss /= world_size
            
    return loss
