#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line argument parser for NeighborRetr.

This module provides function to parse and validate command-line arguments
for the NeighborRetr model training and evaluation.
"""
import argparse


def get_args(description='NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval'):
    """
    Define and parse command-line arguments for NeighborRetr.
    
    Args:
        description (str): Description of the program for the help message
        
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description=description)

    # Loss hyperparameters
    loss_group = parser.add_argument_group('Loss Parameters')
    loss_group.add_argument('--centrality_scale', default=0.3, type=float,
                        help='Scaling factor for centrality weighting')
    loss_group.add_argument('--kl_weight', default=1.0, type=float,
                        help='Weight for KL divergence loss')
    loss_group.add_argument('--uniform_weight', default=1.0, type=float,
                        help='Weight for uniform regularization loss')
    loss_group.add_argument('--ot_temperature', default=0.1, type=float,
                        help='Temperature parameter for optimal transport in uniform regularization')
    loss_group.add_argument('--beta', default=0.7, type=float,
                        help='Beta parameter for uniform regularization loss')
    loss_group.add_argument('--num_neighbors', default=20, type=int,
                        help='Number of neighbors for neighbor adjusting loss')
    loss_group.add_argument('--temperature', default=3.0, type=float,
                        help='Temperature parameter for neighbor adjusting loss')
    loss_group.add_argument('--neighbor_weight', default=1.0, type=float,
                        help='Weight for neighbor adjusting loss')

    # Data loading
    data_group = parser.add_argument_group('Data Loading')
    data_group.add_argument('--workers', default=8, type=int, 
                        help='Number of data loading workers (default: 8)')
    data_group.add_argument('--pin_memory', action='store_true',
                        help='Use pinned memory for faster GPU transfer')
    data_group.add_argument('--prefetch_factor', default=4, type=int,
                        help='Number of batches prefetched by each worker')
    data_group.add_argument('--persistent_workers', action='store_true',
                        help='Keep worker processes alive after dataset exhaustion')
    data_group.add_argument('--video_cache_size', default=64, type=int,
                        help='Size of video frame cache (number of videos)')
    data_group.add_argument('--use_prefetch', action='store_true',
                        help='Enable background prefetching of videos')
    data_group.add_argument('--timeout', default=0, type=int,
                        help='DataLoader timeout in seconds (0 = disabled)')

    # Training/testing modes
    mode_group = parser.add_argument_group('Training/Testing Modes')
    mode_group.add_argument("--save_model", action='store_true', 
                        help="Whether to save model checkpoints")
    mode_group.add_argument("--do_train", type=int, default=0, 
                        help="Whether to run training")
    mode_group.add_argument("--do_eval", type=int, default=0, 
                        help="Whether to run evaluation")
    mode_group.add_argument("--detect_grad", action='store_true', 
                        help="Whether to detect gradient anomaly")

    # Dataset configuration
    dataset_group = parser.add_argument_group('Dataset Configuration')
    dataset_group.add_argument("--datatype", default="msrvtt", type=str, 
                        help="Dataset type for fine-tuning")
    dataset_group.add_argument('--anno_path', type=str, default="data/MSR-VTT/anno", 
                        help='Path to dataset annotations')
    dataset_group.add_argument('--video_path', type=str, default="data/MSR-VTT/videos", 
                        help='Path to video data')
    dataset_group.add_argument("--output_dir", default="output", type=str,
                        help="Output directory for model checkpoints and predictions")

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')

    # Optimization parameters
    optim_group = parser.add_argument_group('Optimization Parameters')
    optim_group.add_argument('--lr', type=float, default=1e-4, 
                        help='Initial learning rate')
    optim_group.add_argument('--coef_lr', type=float, default=1e-3, 
                        help='Coefficient for BERT branch learning rate')
    optim_group.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training for learning rate warmup")
    optim_group.add_argument('--weight_decay', type=float, default=0.2, 
                        help='Weight decay for optimizer')
    optim_group.add_argument('--epochs', type=int, default=5, 
                        help='Maximum number of training epochs')
    
    # Batch sizes
    batch_group = parser.add_argument_group('Batch Configuration')
    batch_group.add_argument('--batch_size', type=int, default=128, 
                        help='Training batch size')
    batch_group.add_argument('--batch_size_val', type=int, default=128, 
                        help='Validation/testing batch size')
    batch_group.add_argument('--memory_size', type=int, default=512, 
                        help='Size of memory bank for neighbor adjusting')
    batch_group.add_argument('--mb_batch', type=int, default=10, 
                        help='Number of batches to use for memory bank creation (lower values use less memory)')

    # Data preprocessing
    preproc_group = parser.add_argument_group('Data Preprocessing')
    preproc_group.add_argument('--max_words', type=int, default=24, 
                        help='Maximum number of text tokens')
    preproc_group.add_argument('--max_frames', type=int, default=12, 
                        help='Maximum number of video frames')
    preproc_group.add_argument('--video_framerate', type=int, default=1, 
                        help='Frame rate for video sampling')

    # Distributed training
    dist_group = parser.add_argument_group('Distributed Training')
    dist_group.add_argument("--device", default='cpu', type=str, 
                        help="Device for training (cpu/cuda)")
    dist_group.add_argument("--world_size", default=1, type=int, 
                        help="World size for distributed training")
    dist_group.add_argument("--local_rank", default=0, type=int, 
                        help="Local rank for distributed training")
    dist_group.add_argument("--distributed", default=0, type=int, 
                        help="Whether to use multi-machine distributed training")

    # Logging and model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--n_display', type=int, default=50, 
                        help='Information display frequency during training')
    model_group.add_argument("--base_encoder", default="ViT-B/32", type=str, 
                        help="CLIP vision encoder backbone")
    model_group.add_argument('--num_hidden_layers', type=int, default=4,
                        help='Number of transformer layers for video feature aggregation')
    model_group.add_argument("--init_model", default=None, type=str, required=False, 
                        help="Path to initial model checkpoint")
    
    args = parser.parse_args()

    # Validate arguments
    _validate_args(args)

    return args


def _validate_args(args):
    """
    Validate parsed arguments for consistency.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments
        
    Raises:
        ValueError: If arguments are invalid or inconsistent
    """
    if args.do_train and args.batch_size % args.world_size != 0:
        raise ValueError(
            f"Invalid batch_size/world_size parameter: {args.batch_size}%{args.world_size} should be == 0")
            
    if args.do_train and args.batch_size_val % args.world_size != 0:
        raise ValueError(
            f"Invalid batch_size_val/world_size parameter: {args.batch_size_val}%{args.world_size} should be == 0")