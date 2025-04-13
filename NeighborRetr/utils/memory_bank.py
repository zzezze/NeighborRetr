#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Memory bank management for NeighborRetr.

This module provides functionality to create, load, and manage the memory bank
used for neighbor adjustment during training. The memory bank stores features
extracted from a subset of training samples to calculate the Neighbor Adjusting Loss.

Reference: "NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval"
"""
import os
import gc
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from NeighborRetr.models.until_module import AllGather

allgather = AllGather.apply


class MemoryBankManager:
    """
    Manager for the memory bank used in NeighborRetr training.
    
    The memory bank stores features from a subset of the training dataset
    to efficiently compute the centrality scores and adjust neighbors
    during cross-modal retrieval training.
    """
    
    def __init__(self, args):
        """
        Initialize the memory bank manager.
        
        Args:
            args: Configuration arguments containing:
                - logger: Logger instance for logging messages
                - mb_batch: Number of batches to include in memory bank
                - batch_size: Batch size for training
                - device: Device to use for computation
        """
        self.args = args
        self.logger = args.logger
        # Use mb_batch parameter for memory bank size (number of batches)
        self.mb_batch = getattr(args, 'mb_batch', 10)  # Default to 10 if not specified
        self.batch_size = args.batch_size
        self.memory_bank_dataloader = None

    def create_memory_bank_dataloader(self):
        """
        Create a dataloader specifically for loading the memory bank.
        
        Returns:
            DataLoader: The memory bank dataloader
        """
        from argparse import Namespace
        from NeighborRetr.dataloaders.data_dataloaders import DATALOADER_DICT
        from NeighborRetr.models.tokenization_clip import SimpleTokenizer as ClipTokenizer
        
        # Create memory bank dataloader using the same configuration as training
        memory_bank_args = Namespace(**vars(self.args))
        
        # Create tokenizer
        tokenizer = ClipTokenizer()
        
        # Create the memory bank dataloader using the same function as training dataloader
        if self.args.datatype in DATALOADER_DICT and DATALOADER_DICT[self.args.datatype]["train"] is not None:
            memory_bank_dataloader, _, memory_bank_sampler = DATALOADER_DICT[self.args.datatype]["train"](
                memory_bank_args, tokenizer)
            
            self.logger.info(f"Created memory bank dataloader with batch size {self.batch_size}")
            self.logger.info(f"Memory bank will use up to {self.mb_batch} batches")
            
            self.memory_bank_dataloader = memory_bank_dataloader
            return memory_bank_dataloader
        else:
            self.logger.error(f"Cannot create memory bank dataloader: datatype {self.args.datatype} not found")
            return None

    def load_memory_bank(self, model, memory_bank_dataloader, device, epoch):
        """
        Load features into the memory bank from the dataset.
        
        This method extracts features from a subset of the dataset to populate
        the model's memory bank, which is used for neighbor adjusting during training.
        
        Args:
            model: Neural network model
            memory_bank_dataloader: DataLoader for memory bank samples
            device: Computing device
            epoch: Current training epoch
            
        Returns:
            int: The actual size of the memory bank (number of samples)
        """
        # Ensure model is in eval mode for feature extraction
        original_model = model
        if hasattr(model, 'module'):
            model = model.module.to(device)
        else:
            model = model.to(device)
        model.eval()
        
        # Initialize lists to store features and masks
        feature_lists = {
            'indices': [],
            'text_features': [],
            'text_masks': [],
            'video_features': [],
            'video_masks': []
        }
        
        # Use the provided memory bank dataloader or the stored one
        if memory_bank_dataloader is None:
            memory_bank_dataloader = self.memory_bank_dataloader
            if memory_bank_dataloader is None:
                self.logger.error("No memory bank dataloader available. Creating one now.")
                memory_bank_dataloader = self.create_memory_bank_dataloader()
                if memory_bank_dataloader is None:
                    self.logger.error("Failed to create memory bank dataloader")
                    return 0
        
        # Determine how many batches to process for the memory bank
        target_batches = min(self.mb_batch, len(memory_bank_dataloader))
        self.logger.info(f"Memory bank loading: target {target_batches} batches out of {len(memory_bank_dataloader)}")
        
        # Extract features from memory bank batches
        with torch.no_grad():
            # Use a context manager to optimize CUDA memory usage
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                for batch_idx, batch in enumerate(tqdm(memory_bank_dataloader, total=target_batches, 
                                                    desc=f"Memory bank (epoch {epoch})")):
                    # Stop after processing the target number of batches
                    if batch_idx >= target_batches:
                        break
                        
                    # Move batch to device
                    batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
                    text_ids, text_mask, video, video_mask, indices, _ = batch
                    
                    # Extract features
                    text_feat, video_feat = model.get_text_video_feat(text_ids, text_mask, video, video_mask)
                    
                    # Store extracted features and masks
                    feature_lists['indices'].append(indices)
                    feature_lists['text_features'].append(text_feat)
                    feature_lists['text_masks'].append(text_mask)
                    feature_lists['video_features'].append(video_feat)
                    feature_lists['video_masks'].append(video_mask)
                    
                    # Log memory usage periodically
                    if batch_idx % 5 == 0 and batch_idx > 0 and torch.cuda.is_available():
                        memory_usage = torch.cuda.memory_allocated() / (1024 ** 3)
                        self.logger.info(f"Memory bank loading progress: {batch_idx}/{target_batches} batches, "
                                    f"GPU memory: {memory_usage:.2f} GB")
                    
                    # Free up memory after each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Log the number of batches processed
        num_batches_processed = len(feature_lists['indices'])
        self.logger.info(f"Processed {num_batches_processed} batches for memory bank")
        
        # If no batches were processed, return 0
        if num_batches_processed == 0:
            self.logger.warning("No batches were processed for the memory bank")
            return 0
        
        # Concatenate features from all batches
        memory_bank_tensors = {}
        try:
            # First concatenate local tensors
            local_tensors = {
                'indices': torch.cat(feature_lists['indices'], dim=0).squeeze(),
                'text_features': torch.cat(feature_lists['text_features'], dim=0),
                'text_masks': torch.cat(feature_lists['text_masks'], dim=0),
                'video_features': torch.cat(feature_lists['video_features'], dim=0),
                'video_masks': torch.cat(feature_lists['video_masks'], dim=0)
            }
            
            # Then gather from all processes if distributed training is enabled
            if self.args.distributed and torch.cuda.is_available():
                memory_bank_tensors = {
                    'indices': allgather(local_tensors['indices'], self.args).squeeze(),
                    'text_features': allgather(local_tensors['text_features'], self.args),
                    'text_masks': allgather(local_tensors['text_masks'], self.args),
                    'video_features': allgather(local_tensors['video_features'], self.args),
                    'video_masks': allgather(local_tensors['video_masks'], self.args)
                }
            else:
                memory_bank_tensors = local_tensors
        except (RuntimeError, ValueError) as e:
            self.logger.error(f"Error during tensor concatenation or gathering: {e}")
            self.logger.warning("Using only local features for memory bank")
            # Use a simpler approach as fallback
            memory_bank_tensors = {
                'indices': torch.cat(feature_lists['indices'], dim=0).squeeze(),
                'text_features': torch.cat(feature_lists['text_features'], dim=0),
                'text_masks': torch.cat(feature_lists['text_masks'], dim=0),
                'video_features': torch.cat(feature_lists['video_features'], dim=0),
                'video_masks': torch.cat(feature_lists['video_masks'], dim=0)
            }
        
        # Update model's memory bank
        model.mb_ind = memory_bank_tensors['indices']
        model.mb_feat_t = memory_bank_tensors['text_features']
        model.mb_mask_t = memory_bank_tensors['text_masks']
        model.mb_feat_v = memory_bank_tensors['video_features']
        model.mb_mask_v = memory_bank_tensors['video_masks']
        model.mb_batch = memory_bank_tensors['text_features'].size(0)
        
        # Log memory bank statistics
        memory_bank_size = memory_bank_tensors['text_features'].size(0)
        text_feature_shape = memory_bank_tensors['text_features'].shape
        video_feature_shape = memory_bank_tensors['video_features'].shape
        
        # Calculate memory usage of memory bank tensors
        memory_usage_bytes = sum(tensor.numel() * tensor.element_size() 
                                for tensor in memory_bank_tensors.values())
        memory_usage_gb = memory_usage_bytes / (1024 ** 3)
        
        self.logger.info(f"Memory bank size: {memory_bank_size} samples, {memory_usage_gb:.3f} GB")
        self.logger.info(f"Feature dimensions - Text: {text_feature_shape}, Video: {video_feature_shape}")
        
        # Restore the original model reference
        model = original_model
        
        return memory_bank_size

    def clear_memory_bank(self, model):
        """
        Clear the memory bank to free up GPU memory.
        
        This method should be called at the end of each epoch to ensure
        memory is properly managed, especially for large datasets.
        
        Args:
            model: Neural network model with memory bank attributes
            
        Returns:
            model: The model with cleared memory bank
        """
        if hasattr(model, 'module'):
            target_model = model.module
        else:
            target_model = model
            
        # Log memory bank size before clearing
        if hasattr(target_model, 'mb_batch') and target_model.mb_batch > 0:
            self.logger.info(f"Clearing memory bank with {target_model.mb_batch} samples")
            
        # Clear memory bank attributes
        device = next(target_model.parameters()).device
        target_model.mb_ind = torch.tensor([], dtype=torch.long, device=device)
        target_model.mb_feat_t = torch.empty((0, 0, 0), dtype=torch.float, device=device)
        target_model.mb_feat_v = torch.empty((0, 0, 0), dtype=torch.float, device=device)
        target_model.mb_mask_t = torch.empty((0, 0), dtype=torch.float, device=device)
        target_model.mb_mask_v = torch.empty((0, 0), dtype=torch.float, device=device)
        target_model.mb_batch = 0
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        self.logger.info("Memory bank cleared")
        return model