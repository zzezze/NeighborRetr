#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval
Main entry point script for training and evaluation.

This script serves as the entry point for the NeighborRetr model, handling model building,
data loading, training, and evaluation according to command line arguments.

Reference:
"NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval"
"""
from __future__ import absolute_import, division, unicode_literals, print_function

# add warning filter
import warnings
# ignore all warnings
warnings.filterwarnings("ignore")


import os
import time
import torch
import gc
from datetime import datetime, timedelta, timezone
from argparse import Namespace

from NeighborRetr.config.args_parser import get_args
from NeighborRetr.utils.setup import set_seed_logger, setup_distributed_environment
from NeighborRetr.training.trainer import train_epoch
from NeighborRetr.training.evaluator import eval_epoch
from NeighborRetr.training.optimizer import prep_optimizer
from NeighborRetr.utils.memory_bank import MemoryBankManager
from NeighborRetr.utils.metric_logger import MetricLogger
from NeighborRetr.utils.metrics import RetrievalMetrics

from NeighborRetr.dataloaders.data_dataloaders import DATALOADER_DICT
from NeighborRetr.models.tokenization_clip import SimpleTokenizer as ClipTokenizer
from NeighborRetr.models.modeling import NeighborRetr
from NeighborRetr.utils.comm import is_main_process, synchronize
from NeighborRetr.utils.logger import setup_logger


def build_model(args):
    """
    Build the NeighborRetr model and load pretrained weights if specified.
    
    Args:
        args (Namespace): Command line arguments containing model configuration
        
    Returns:
        nn.Module: The initialized NeighborRetr model
        
    Raises:
        FileNotFoundError: If the specified model checkpoint does not exist
    """
    model = NeighborRetr(args)
    
    # Load pre-trained weights if specified
    if args.init_model:
        if not os.path.exists(args.init_model):
            raise FileNotFoundError(f"Model checkpoint not found: {args.init_model}")
            
        args.logger.info(f"Loading pre-trained weights from {args.init_model}")
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        model.load_state_dict(model_state_dict, strict=False)
        args.logger.info("Pre-trained weights loaded successfully")

    # Move model to the specified device
    model.to(args.device)
    return model

def build_dataloader(args):
    """
    Build dataloaders for training, validation, and testing.
    
    Args:
        args (Namespace): Command line arguments containing dataloader configuration
        
    Returns:
        tuple: (test_dataloader, val_dataloader, train_dataloader, train_sampler)
        
    Raises:
        AssertionError: If the specified datatype is not supported or lacks test/val sets
    """
    tokenizer = ClipTokenizer()
    
    # Validate datatype is supported
    assert args.datatype in DATALOADER_DICT, f"Datatype {args.datatype} not supported"
    assert (DATALOADER_DICT[args.datatype]["test"] is not None or 
            DATALOADER_DICT[args.datatype]["val"] is not None), \
           f"Datatype {args.datatype} must have either test or val set"

    logger = args.logger
    logger.info(f"BUILDING DATALOADERS: {args.datatype.upper()}")
    logger.info("=" * 80)

    # Build test dataloader
    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)
        logger.info(f"Test dataloader created successfully")

    # Build validation dataloader (use test if val is not available)
    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
        logger.info(f"Validation dataloader created successfully")
    else:
        val_dataloader, val_length = test_dataloader, test_length
        logger.info(f"Using test dataloader for validation")

    # Report validation results if the test set is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length
        logger.info(f"Using validation dataloader for testing")

    # Build training dataloader if training mode is enabled
    train_dataloader, train_sampler = None, None
    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](
            args, tokenizer)
        logger.info(f"Training dataloader created successfully")

    # Log information about the datasets in a tabular format
    logger.info("=" * 80)
    logger.info("DATASET STATISTICS")
    
    if isinstance(test_length, int):
        # Format as a table for better readability
        logger.info(f"┌{'─' * 30}┬{'─' * 15}┬{'─' * 15}┬{'─' * 15}┐")
        logger.info(f"│ {'Dataset':^30} │ {'Examples':^15} │ {'Batch Size':^15} │ {'Steps':^15} │")
        logger.info(f"├{'─' * 30}┼{'─' * 15}┼{'─' * 15}┼{'─' * 15}┤")
        logger.info(f"│ {'Test Set':^30} │ {test_length:^15} │ {args.batch_size_val:^15} │ {len(test_dataloader):^15} │")
        logger.info(f"│ {'Val Set':^30} │ {val_length:^15} │ {args.batch_size_val:^15} │ {len(val_dataloader):^15} │")
        
        if args.do_train and train_dataloader is not None:  # check if train_dataloader exists
            logger.info(f"│ {'Train Set':^30} │ {train_length:^15} │ {args.batch_size:^15} │ {len(train_dataloader):^15} │")
        
        logger.info(f"└{'─' * 30}┴{'─' * 15}┴{'─' * 15}┴{'─' * 15}┘")
    
    elif len(test_length) == 2:
        # For the case with separate text and video counts
        logger.info(f"┌{'─' * 30}┬{'─' * 25}┬{'─' * 15}┬{'─' * 15}┐")
        logger.info(f"│ {'Dataset':^30} │ {'Examples (T/V)':^25} │ {'Batch Size':^15} │ {'Steps':^15} │")
        logger.info(f"├{'─' * 30}┼{'─' * 25}┼{'─' * 15}┼{'─' * 15}┤")
        logger.info(f"│ {'Test Set':^30} │ {f'{test_length[0]}/{test_length[1]}':^25} │ {args.batch_size_val:^15} │ {f'{len(test_dataloader[0])}/{len(test_dataloader[1])}':^15} │")
        logger.info(f"│ {'Val Set':^30} │ {f'{val_length[0]}/{val_length[1]}':^25} │ {args.batch_size_val:^15} │ {f'{len(val_dataloader[0])}/{len(val_dataloader[1])}':^15} │")
        
        if args.do_train and train_dataloader is not None:  # check if train_dataloader exists
            logger.info(f"│ {'Train Set':^30} │ {train_length:^25} │ {args.batch_size:^15} │ {len(train_dataloader):^15} │")
        
        logger.info(f"└{'─' * 30}┴{'─' * 25}┴{'─' * 15}┴{'─' * 15}┘")
    
    if args.do_train and train_dataloader is not None:
        logger.info(f"Total training steps: {len(train_dataloader) * args.epochs}")

    logger.info("=" * 80)
    
    return test_dataloader, val_dataloader, train_dataloader, train_sampler

def save_model(epoch, args, model, type_name=""):
    """
    Save model checkpoint to file.
    
    Args:
        epoch (int): Current epoch number
        args (Namespace): Command line arguments
        model (nn.Module): Model to save
        type_name (str, optional): Type suffix for the checkpoint file name
        
    Returns:
        str: Path to the saved model file
    """
    # Only save the model itself, not the distributed wrapper
    model_to_save = model.module if hasattr(model, 'module') else model
    
    # Create checkpoint filename with optional type name
    output_model_file = os.path.join(
        args.output_dir, 
        f"pytorch_model.bin.{type_name + '.' if type_name else ''}{epoch}"
    )
    
    # Save model state dictionary
    torch.save(model_to_save.state_dict(), output_model_file)
    args.logger.info("Model saved to %s", output_model_file)
    
    return output_model_file

def main():
    """
    Main function that orchestrates the entire training and evaluation process.
    
    This function handles:
    1. Argument parsing and setup
    2. Directory and logging configuration
    3. Model building and data loading
    4. Training loop with memory bank management
    5. Model evaluation and saving
    """
    # Parse arguments and set up environment
    args = get_args()
    
    # Configure output directories with timestamped folder
    base_output_dir = args.output_dir
    datatype_dir = os.path.join(base_output_dir, args.datatype)
    args.output_dir = datatype_dir

    # Create timestamped folder for this run (Beijing timezone = UTC+8)
    current_time_beijing = datetime.now(timezone(timedelta(hours=8))).strftime("%m-%d-%H-%M")
    run_dir = os.path.join(args.output_dir, current_time_beijing)
    os.makedirs(run_dir, exist_ok=True)
    args.output_dir = run_dir

    # Set up logger and random seeds
    args.logger = setup_logger('tvr', args.output_dir, args.local_rank)
    args = set_seed_logger(args)
    
    # Set up distributed environment if needed
    args = setup_distributed_environment(args)
    
    # Initialize metric logger
    meters = MetricLogger(delimiter="  ")
    
    # Initialize RetrievalMetrics tracker
    metrics_tracker = RetrievalMetrics(logger=args.logger)
    
    # Print welcome message and configuration summary
    if is_main_process():
        args.logger.info("=" * 80)
        args.logger.info("NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval")
        args.logger.info("=" * 80)
        args.logger.info(f"Running with configuration:")
        
        # Print key configuration parameters in a more readable format
        config_items = [
            ("Dataset", args.datatype),
            ("Training", "Yes" if args.do_train else "No"),
            ("Evaluation", "Yes" if args.do_eval else "No"),
            ("Epochs", args.epochs),
            ("Batch Size", args.batch_size),
            ("Learning Rate", args.lr),
            ("Base Encoder", args.base_encoder),
            ("Device", args.device),
            ("Output Directory", args.output_dir)
        ]
        
        # Format configuration as a table
        max_key_len = max(len(k) for k, _ in config_items)
        for key, value in config_items:
            args.logger.info(f"  {key:<{max_key_len+2}}: {value}")
        
        args.logger.info("=" * 80)
    
    # Build model and dataloaders
    model = build_model(args)
    test_dataloader, val_dataloader, train_dataloader, train_sampler = build_dataloader(args)

    # Training process
    if args.do_train:
        # 检查训练数据加载器是否有效
        if train_dataloader is None:
            args.logger.error("Failed to create training dataloader. Exiting.")
            return
            
        # Initialize Memory Bank Manager
        memory_bank_manager = MemoryBankManager(args)
        
        # Create memory bank dataloader
        memory_bank_dataloader = memory_bank_manager.create_memory_bank_dataloader()
        
        # Setup training start time tracking
        training_start_time = time.time()
        
        # Calculate max steps and optimizer steps
        max_steps = len(train_dataloader) * args.epochs
        scheduler_steps_ratio = args.epochs
        optimizer_steps = len(train_dataloader) * scheduler_steps_ratio
        
        # Prepare optimizer, scheduler, and model
        optimizer, scheduler, model = prep_optimizer(args, model, optimizer_steps, args.local_rank)

        global_step = 0
        best_score = 0
        best_epoch = 0
        
        # Training loop
        for epoch in range(args.epochs):
            if is_main_process():
                args.logger.info("=" * 80)
                args.logger.info(f"STARTING EPOCH {epoch+1}/{args.epochs}")
                args.logger.info("=" * 80)
            
            # Set epoch for distributed sampler
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            synchronize()
            
            # Clear GPU cache before each epoch
            torch.cuda.empty_cache()

            # Load memory bank for this epoch
            start_time = time.time()
            if is_main_process():
                args.logger.info(f"Loading memory bank for epoch {epoch+1}")
            
            memory_bank_size = memory_bank_manager.load_memory_bank(
                model, 
                memory_bank_dataloader, 
                args.device, 
                epoch
            )
            
            elapsed_time_minutes = (time.time() - start_time) / 60
            if is_main_process():
                args.logger.info(f"Memory bank loaded in {elapsed_time_minutes:.2f} minutes")
                args.logger.info(f"Memory bank size: {memory_bank_size} samples")

            # Train for one epoch
            train_loss, global_step, best_t2v_metrics, best_v2t_metrics = train_epoch(
                epoch, args, model, train_dataloader, 
                args.device, args.world_size, optimizer,
                scheduler, global_step, max_steps, val_dataloader,
                meters
            )
            
            # Evaluate on full validation set
            torch.cuda.empty_cache()
            if is_main_process():
                args.logger.info("=" * 80)
                args.logger.info(f"EVALUATING MODEL AFTER EPOCH {epoch+1}/{args.epochs}")
                args.logger.info("=" * 80)
            
            t2v_metrics, v2t_metrics = eval_epoch(args, model, val_dataloader, args.device)

            # Update best metrics
            is_updated = False
            if args.local_rank == 0:
                # Calculate average R@1 score
                current_score = (t2v_metrics['R1'] + v2t_metrics['R1']) / 2
                
                is_updated, _ = metrics_tracker.update_best_metrics(
                    t2v_metrics, 
                    v2t_metrics, 
                    t2v_metrics['R1'], 
                    v2t_metrics['R1']
                )
                
                # If metrics were updated, log best metrics
                if is_updated:
                    metrics_tracker.log_best_metrics()
                    best_score = current_score
                    best_epoch = epoch
                
                # Save model checkpoint if this is the main process
                if args.save_model:
                    output_model_file = save_model(epoch, args, model)
                    
                    # Save best model
                    if current_score >= best_score:
                        best_model_path = os.path.join(args.output_dir, 'best.pth')
                        torch.save(
                            model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                            best_model_path
                        )
                        args.logger.info(f"New best model saved to {best_model_path} with R@1 score: {current_score:.4f}")

            # Clear memory bank data before next epoch
            model = memory_bank_manager.clear_memory_bank(model)
            
            # Force garbage collection
            torch.cuda.empty_cache()
            gc.collect()

            synchronize()
        
        # Report total training time
        training_time_seconds = time.time() - training_start_time
        training_time_formatted = time.strftime("%Hh %Mmin %Ss", time.gmtime(training_time_seconds))
        
        if is_main_process():
            args.logger.info("=" * 80)
            args.logger.info(f"TRAINING COMPLETE")
            args.logger.info("=" * 80)
            args.logger.info(f"Total training time: {training_time_formatted}")
            args.logger.info(f"Best model achieved at epoch {best_epoch+1} with R@1 score: {best_score:.4f}")
            args.logger.info("=" * 80)

        # Test on the best checkpoint
        if args.save_model and os.path.exists(os.path.join(args.output_dir, 'best.pth')):
            if is_main_process():
                args.logger.info("=" * 80)
                args.logger.info("FINAL EVALUATION ON TEST SET")
                args.logger.info("Loading best model for final evaluation")
                args.logger.info("=" * 80)
            
            model = model.module if hasattr(model, 'module') else model
            model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best.pth'), 
                                           map_location='cpu'), 
                                strict=False)
                                
            # Re-wrap model for distributed training if needed
            if args.distributed and torch.cuda.is_available():
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.local_rank], find_unused_parameters=True
                )

            # Final evaluation on test set
            torch.cuda.empty_cache()
            t2v_metrics, v2t_metrics = eval_epoch(args, model, test_dataloader, args.device)
            
            # Final report of best performance
            if is_main_process():
                best_metrics = metrics_tracker.get_best_metrics()
                final_best_score = (best_metrics['text_to_video']['R1'] + best_metrics['video_to_text']['R1']) / 2
                
                args.logger.info("=" * 80)
                args.logger.info("FINAL BEST RESULTS")
                args.logger.info("=" * 80)
                args.logger.info(f"Final best R1 score: {final_best_score:.4f}")
                
                if best_metrics['text_to_video'] is not None and best_metrics['video_to_text'] is not None:
                    args.logger.info("Best Text-to-Video Results:")
                    metrics_tracker.print_metrics(best_metrics['text_to_video'], prefix="  ")
                    
                    args.logger.info("Best Video-to-Text Results:")
                    metrics_tracker.print_metrics(best_metrics['video_to_text'], prefix="  ")
                args.logger.info("=" * 80)
    
    # Evaluation only mode
    elif args.do_eval:
        if is_main_process():
            args.logger.info("=" * 80)
            args.logger.info("EVALUATION MODE")
            args.logger.info("=" * 80)
        
        eval_epoch(args, model, test_dataloader, args.device)


if __name__ == "__main__":
    main()