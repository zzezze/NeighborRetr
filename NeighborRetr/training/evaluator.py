#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation functionality for NeighborRetr.

This module provides functions to evaluate the performance of the NeighborRetr model
on text-to-video and video-to-text retrieval tasks.
"""
import time
import torch
import numpy as np
import logging  # 添加这一行导入logging模块
from tqdm import tqdm
from NeighborRetr.utils.metrics import RetrievalMetrics
from NeighborRetr.models.modeling import AllGather
from NeighborRetr.utils.comm import is_main_process

allgather = AllGather.apply


def _run_on_single_gpu(model, t_mask_list, v_mask_list, t_feat_list, v_feat_list, mini_batch=64):
    """
    Compute similarity matrices on a single GPU by processing in mini-batches.
    
    Args:
        model (nn.Module): Model to evaluate
        t_mask_list (torch.Tensor): Text attention masks
        v_mask_list (torch.Tensor): Video attention masks
        t_feat_list (torch.Tensor): Text features
        v_feat_list (torch.Tensor): Video features
        mini_batch (int): Batch size for processing
        
    Returns:
        tuple: (text_to_video_sim, video_to_text_sim)
    """
    logger = getattr(model, 'logger', None)
    
    # Split data into mini-batches to avoid OOM
    batch_t_mask = torch.split(t_mask_list, mini_batch)
    batch_v_mask = torch.split(v_mask_list, mini_batch)
    batch_t_feat = torch.split(t_feat_list, mini_batch)
    batch_v_feat = torch.split(v_feat_list, mini_batch)
 
    sim_matrix = []
        
    # Process each mini-batch combination
    with torch.no_grad():
        for idx1, (t_mask, t_feat) in enumerate(zip(batch_t_mask, batch_t_feat)):
            each_row = []
            for idx2, (v_mask, v_feat) in enumerate(zip(batch_v_mask, batch_v_feat)):
                # Get similarity scores between text and video features
                logits, _, *_tmp = model.get_similarity_logits(t_feat, v_feat, t_mask, v_mask)
                logits = logits.cpu().detach().numpy()
                each_row.append(logits)
            # Concatenate all video results for this text batch
            each_row = np.concatenate(tuple(each_row), axis=-1)
            sim_matrix.append(each_row)

    # Concatenate all text batches to get the final similarity matrix
    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    # Return both similarity matrices (t2v and v2t are just transposes of each other)
    return sim_matrix, sim_matrix.T


def eval_epoch(args, model, test_dataloader, device):
    """
    Evaluate the model on the test set.
    
    Args:
        args (argparse.Namespace): Command line arguments
        model (nn.Module): Model to evaluate
        test_dataloader (DataLoader): DataLoader for test data
        device (torch.device): Computing device
        
    Returns:
        tuple: (text_to_video_metrics, video_to_text_metrics)
            - text_to_video_metrics (dict): Text-to-video retrieval metrics
            - video_to_text_metrics (dict): Video-to-text retrieval metrics
    """
    logger = args.logger
    metrics_tracker = RetrievalMetrics(logger=logger)
    
    # Ensure we're evaluating the model, not the DDP wrapper
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # Check if we're using multiple sentences per video
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_ and is_main_process():
        logger.info("Evaluating with multi-sentence per video setup")
        logger.info(f"Sentences: {sentence_num_}, Videos: {video_num_}")

    # Switch to evaluation mode
    model.eval()
    
    # ----------------------------
    # 1. Cache the features
    # ----------------------------
    batch_mask_t, batch_mask_v, batch_feat_t, batch_feat_v, ids_t, ids_v = [], [], [], [], [], []

    with torch.no_grad():
        tic = time.time()
        if multi_sentence_:  # Multi-sentences retrieval processing
            total_video_num = 0
            if is_main_process():
                logger.info('Extracting text and video features...')
                
            for batch in tqdm(test_dataloader):
                batch = tuple(t.to(device) for t in batch)
                text_ids, text_mask, video, video_mask, inds, _ = batch

                b, *_t = video.shape
                text_feat = model.get_text_feat(text_ids, text_mask)
                ids_t.append(inds)
                batch_mask_t.append(text_mask)
                batch_feat_t.append(text_feat)

                video_feat = model.get_video_feat(video, video_mask)
                batch_mask_v.append(video_mask)
                batch_feat_v.append(video_feat)

                total_video_num += b

            if is_main_process():
                logger.info('Processing collected features...')
                
            ids_t = torch.cat(ids_t, dim=0).squeeze()
            batch_mask_t = torch.cat(batch_mask_t, dim=0)
            batch_mask_v = torch.cat(batch_mask_v, dim=0)
            batch_feat_t = torch.cat(batch_feat_t, dim=0)
            batch_feat_v = torch.cat(batch_feat_v, dim=0)

            # Filter video features based on cut-off points
            cut_off_points_tensor = torch.tensor(cut_off_points_, device=ids_t.device)
            # Create a boolean mask
            mask = torch.zeros_like(ids_t, dtype=torch.bool)

            # Set True for positions matching cut-off points
            for point in cut_off_points_tensor:
                mask |= (ids_t == point)

            # Use boolean indexing to select matching video features and masks
            batch_feat_v = batch_feat_v[mask]
            batch_mask_v = batch_mask_v[mask]
        else:
            # Standard single-sentence per video processing
            if is_main_process():
                logger.info('Extracting text and video features...')

            for batch in tqdm(test_dataloader):
                batch = tuple(t.to(device) for t in batch)
                text_ids, text_mask, video, video_mask, inds, _ = batch
                video_mask = video_mask.view(-1, video_mask.shape[-1])
                text_feat, video_feat = model.get_text_video_feat(text_ids, text_mask, video, video_mask)
                ids_t.append(inds)
                batch_mask_t.append(text_mask)
                batch_mask_v.append(video_mask)
                batch_feat_t.append(text_feat)
                batch_feat_v.append(video_feat)

            ids_t = allgather(torch.cat(ids_t, dim=0), args).squeeze()
            batch_mask_t = allgather(torch.cat(batch_mask_t, dim=0), args)
            batch_mask_v = allgather(torch.cat(batch_mask_v, dim=0), args)
            batch_feat_t = allgather(torch.cat(batch_feat_t, dim=0), args)
            batch_feat_v = allgather(torch.cat(batch_feat_v, dim=0), args)

            # Reorder tensors based on indices to ensure consistency
            batch_mask_t[ids_t] = batch_mask_t.clone()
            batch_mask_v[ids_t] = batch_mask_v.clone()
            batch_feat_t[ids_t] = batch_feat_t.clone()
            batch_feat_v[ids_t] = batch_feat_v.clone()
            
            # Trim tensors to valid range
            batch_mask_t = batch_mask_t[:ids_t.max() + 1, ...]
            batch_mask_v = batch_mask_v[:ids_t.max() + 1, ...]
            batch_feat_t = batch_feat_t[:ids_t.max() + 1, ...]
            batch_feat_v = batch_feat_v[:ids_t.max() + 1, ...]

    toc1 = time.time()
    feature_time = toc1 - tic
    
    # 使用安全的调试日志方式
    # Use a safer way to log debug information
    if logger and hasattr(logger, 'isEnabledFor') and is_main_process():
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Feature shapes: Text {batch_feat_t.shape}, Video {batch_feat_v.shape}')

    # ----------------------------------
    # 2. Calculate the similarity
    # ----------------------------------
    if is_main_process():
        logger.info('Calculating similarity matrices...')
        
    with torch.no_grad():
        new_t2v_matrix, new_v2t_matrix = _run_on_single_gpu(
            model, batch_mask_t, batch_mask_v, batch_feat_t, batch_feat_v
        )
        sim_matrix = new_t2v_matrix

    toc2 = time.time()
    similarity_time = toc2 - toc1
    
    # Process similarity matrices differently for multi-sentence vs standard cases
    if multi_sentence_:
        # For multi-sentence, reshape matrices to account for multiple descriptions per video
        new_v2t_matrix = new_v2t_matrix.T
        
        # 更安全的调试日志
        # Safer debug logging
        if logger and hasattr(logger, 'isEnabledFor') and is_main_process():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Similarity matrix before reshape: {sim_matrix.shape[0]} x {sim_matrix.shape[1]}")
        
        # Convert cut-off points to lengths
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)])
        new_t2v_matrix_new, new_v2t_matrix_new = [], []
        
        # Reshape matrices to uniform size by padding with -inf
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            new_t2v_matrix_new.append(np.concatenate((new_t2v_matrix[s_:e_],
                                                    np.full((max_length - e_ + s_, sim_matrix.shape[1]), -np.inf)),
                                                    axis=0))
            new_v2t_matrix_new.append(np.concatenate((new_v2t_matrix[s_:e_],
                                                    np.full((max_length - e_ + s_, sim_matrix.shape[1]), -np.inf)),
                                                    axis=0))
        new_t2v_matrix_new = np.stack(tuple(new_t2v_matrix_new), axis=0)
        new_v2t_matrix_new = np.stack(tuple(new_v2t_matrix_new), axis=0)

        # 更安全的调试日志
        # Safer debug logging
        if logger and hasattr(logger, 'isEnabledFor') and is_main_process():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Similarity matrix after reshape: {new_t2v_matrix_new.shape}")

        # Calculate metrics for reshaped matrices
        new_tv_metrics = metrics_tracker.tensor_text_to_video_metrics(new_t2v_matrix_new)
        v2t_sim = metrics_tracker.tensor_video_to_text_sim(new_v2t_matrix_new)
        new_vt_metrics = metrics_tracker.compute_metrics(v2t_sim)
    else:
        # Standard evaluation
        # 更安全的调试日志
        # Safer debug logging
        if logger and hasattr(logger, 'isEnabledFor') and is_main_process():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Similarity matrix size: {sim_matrix.shape[0]} x {sim_matrix.shape[1]}")
        new_tv_metrics = metrics_tracker.compute_metrics(new_t2v_matrix)
        new_vt_metrics = metrics_tracker.compute_metrics(new_v2t_matrix)

    toc3 = time.time()
    metrics_time = toc3 - toc2

    # Log timing information only on main process
    if is_main_process():
        logger.info(f"Evaluation timing breakdown:")
        logger.info(f"  - Feature extraction: {feature_time:.2f}s")
        logger.info(f"  - Similarity computation: {similarity_time:.2f}s")
        logger.info(f"  - Metrics calculation: {metrics_time:.2f}s")
        logger.info(f"  - Total time: {toc3 - tic:.2f}s")

        # Get retrieval evaluation metrics
        text_to_video_metrics = new_tv_metrics
        video_to_text_metrics = new_vt_metrics
        
        # Calculate mean R1
        mean_r1 = (text_to_video_metrics['R1'] + video_to_text_metrics['R1']) / 2
        
        # Log results in a clean format
        logger.info("=" * 80)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Mean R@1: {mean_r1:.4f}")
        
        logger.info("Text-to-Video Retrieval:")
        metrics_tracker.print_metrics(text_to_video_metrics, prefix="  ")
        
        logger.info("Video-to-Text Retrieval:")
        metrics_tracker.print_metrics(video_to_text_metrics, prefix="  ")
    
    return new_tv_metrics, new_vt_metrics