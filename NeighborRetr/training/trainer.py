#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training functionality for NeighborRetr.

This module provides functions to train the NeighborRetr model for one epoch.
"""
import time
import torch
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from NeighborRetr.utils.setup import reduce_loss
from NeighborRetr.training.evaluator import eval_epoch
from NeighborRetr.utils.comm import is_main_process
from NeighborRetr.utils.metrics import RetrievalMetrics

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                scheduler, global_step, max_steps, val_dataloader, meters):
    """
    Train the model for one epoch.
    
    Args:
        epoch (int): Current epoch number
        args (argparse.Namespace): Command line arguments
        model (nn.Module): Model to train
        train_dataloader (DataLoader): DataLoader for training data
        device (torch.device): Computing device
        n_gpu (int): Number of GPUs
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        global_step (int): Global step counter
        max_steps (int): Maximum number of steps
        val_dataloader (DataLoader): DataLoader for validation
        meters (MetricLogger): MetricLogger for tracking metrics
        
    Returns:
        tuple: (total_loss, global_step, best_t2v_metrics, best_v2t_metrics)
            - total_loss (float): Average loss over the epoch
            - global_step (int): Updated global step
            - best_t2v_metrics (dict): Best text-to-video metrics
            - best_v2t_metrics (dict): Best video-to-text metrics
    """
    logger = args.logger
    
    # Initialize metrics tracker
    metrics_tracker = RetrievalMetrics(logger=logger)
    
    # Clear GPU cache before training
    torch.cuda.empty_cache()
    model.train()
    
    # Configure logging frequency
    log_step = args.n_display
    total_loss = 0
    
    end = time.time()
    logit_scale = 0
    
    # Progress bar for training
    train_iter = enumerate(train_dataloader, start=1)
    if is_main_process():
        train_iter = tqdm(train_iter, total=len(train_dataloader), 
                          desc=f"Epoch {epoch}/{args.epochs}")
    
    for step, (iteration, batch) in enumerate(train_iter):
        global_step += 1
        data_time = time.time() - end

        # Move batch to device
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        text_ids, text_mask, video, video_mask, inds, idx = batch

        # Handle mask dimensions
        if text_mask.dim() == 3:
            text_mask.squeeze(1)
        if video_mask.dim() == 3:
            video_mask.squeeze(1)

        # Forward pass
        loss, centrality_loss, uniform_loss, neighbor_loss, kl_loss = model(
            text_ids, text_mask, video, video_mask, idx, global_step, logger
        )

        # Handle multi-GPU case
        if n_gpu > 1:
            loss = loss.mean()
            centrality_loss = centrality_loss.mean()
            uniform_loss = uniform_loss.mean()
            neighbor_loss = neighbor_loss.mean()
            kl_loss = kl_loss.mean()

        # Backward pass
        if args.detect_grad:
            with torch.autograd.detect_anomaly():
                loss.backward()
        else:
            loss.backward()

        # Gradient clipping and optimization step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()  # Update learning rate schedule
        
        optimizer.zero_grad()

        # Clamp logit scale to prevent numerical instability
        # https://github.com/openai/CLIP/issues/46
        if hasattr(model, 'module'):
            torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.module.clip.logit_scale.exp().item()
        else:
            torch.clamp_(model.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.clip.logit_scale.exp().item()

        batch_time = time.time() - end
        end = time.time()

        # Reduce losses across processes for logging
        reduced_l = reduce_loss(loss, args)
        reduced_centrality_loss = reduce_loss(centrality_loss, args)
        reduced_uniform_loss = reduce_loss(uniform_loss, args)
        reduced_neighbor_loss = reduce_loss(neighbor_loss, args)
        reduced_kl_loss = reduce_loss(kl_loss, args)

        # Update metrics
        meters.update(
            time=batch_time, 
            data=data_time, 
            loss=float(reduced_l), 
            centrality_loss=float(reduced_centrality_loss),
            uniform_loss=float(reduced_uniform_loss),
            neighbor_loss=float(reduced_neighbor_loss),
            kl_loss=float(reduced_kl_loss)
        )

        # Calculate ETA
        eta_seconds = meters.time.global_avg * (max_steps - global_step)
        eta_string = str(timedelta(seconds=int(eta_seconds)))

        # Log training progress
        if (global_step % log_step == 0 or global_step == 1) and is_main_process():
            progress_info = [
                f"Epoch: {epoch}/{args.epochs}",
                f"Iter: {global_step}/{max_steps}",
                f"Loss: {meters.loss.median:.4f}",
                f"C-Loss: {meters.centrality_loss.median:.4f}",
                f"U-Loss: {meters.uniform_loss.median:.4f}",
                f"N-Loss: {meters.neighbor_loss.median:.4f}",
                f"KL-Loss: {meters.kl_loss.median:.4f}",
                f"LR: {optimizer.get_lr()[0]:.8f}",
                f"LogitScale: {logit_scale:.2f}",
                f"ETA: {eta_string}"
            ]
            
            logger.info(" | ".join(progress_info))
            
            # Log memory usage less frequently to avoid cluttering logs
            if global_step % (log_step * 5) == 0:
                logger.info(f"Memory usage: {torch.cuda.max_memory_allocated() / 1024.0 / 1024.0:.1f} MB")

        # Periodic validation
        stored_log_step = log_step
        if global_step % (log_step * 3) == 0 or global_step == 1:
            if is_main_process():
                logger.info("=" * 80)
                logger.info(f"Running validation at step {global_step}")
            
            # Evaluate model performance on validation set
            t2v_metrics, v2t_metrics = eval_epoch(
                args, model, val_dataloader, args.device
            )

            # Update best scores only on main process
            if args.local_rank == 0:
                # Use metrics tracker to update best scores
                is_updated, current_score = metrics_tracker.update_best_metrics(
                    t2v_metrics, v2t_metrics, t2v_metrics['R1'], v2t_metrics['R1']
                )
                
                # Only log best metrics if there was an improvement
                if is_updated:
                    metrics_tracker.log_best_metrics()
                    
                    # Save model checkpoint if requested
                    if args.save_model:
                        from main import save_model
                        output_model_file = save_model(epoch, args, model, type_name="best")
                        logger.info(f"New best model saved to: {output_model_file}")
                       
            # Return to training mode
            model.train()
            
            if is_main_process():
                logger.info("=" * 80)
            
        log_step = stored_log_step
        total_loss += loss.item()

    # Calculate average loss over the entire epoch
    total_loss = total_loss / len(train_dataloader)
    
    # Log epoch summary
    if is_main_process():
        logger.info("=" * 80)
        logger.info(f"EPOCH {epoch} SUMMARY")
        logger.info(f"Average loss: {total_loss:.4f}")
        logger.info("=" * 80)
    
    # Get current best metrics
    best_metrics = metrics_tracker.get_best_metrics()
    best_t2v_metrics = best_metrics['text_to_video']
    best_v2t_metrics = best_metrics['video_to_text']

    return (total_loss, global_step, best_t2v_metrics, best_v2t_metrics)