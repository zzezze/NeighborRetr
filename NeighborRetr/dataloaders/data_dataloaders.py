#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data loader factory for NeighborRetr.

This module provides functions to create data loaders for various
video-text datasets used in cross-modal retrieval tasks.
"""

import torch
from torch.utils.data import DataLoader
from .dataloader_msrvtt_retrieval import MSRVTTDataset
from .dataloader_activitynet_retrieval import ActivityNetDataset
from .dataloader_didemo_retrieval import DiDeMoDataset
from .dataloader_msvd_retrieval import MsvdDataset


def _create_dataloader_common(dataset, args, batch_size, is_train=True):
    """
    Common dataloader creation with consistent parameters.
    
    Args:
        dataset: Dataset instance
        args: Configuration arguments
        batch_size: Batch size to use
        is_train: Whether this is a training dataloader
        
    Returns:
        DataLoader instance
    """
    try:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    except:
        sampler = None  # cpu
        
    return DataLoader(
        dataset,
        batch_size=batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=getattr(args, 'pin_memory', False),
        shuffle=(sampler is None and is_train),
        sampler=sampler,
        drop_last=is_train,
        prefetch_factor=getattr(args, 'prefetch_factor', 2) if args.workers > 0 else None,
        persistent_workers=getattr(args, 'persistent_workers', False) if args.workers > 0 else False,
        timeout=getattr(args, 'timeout', 0)
    )


def dataloader_msrvtt_train(args, tokenizer):
    """
    Create training dataloader for MSR-VTT dataset.
    
    Args:
        args: Configuration arguments
        tokenizer: Tokenizer for text processing
        
    Returns:
        tuple: (dataloader, dataset_size, sampler)
    """
    msrvtt_dataset = MSRVTTDataset(
        subset='train',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    except:
        train_sampler = None  # cpu
        
    dataloader = _create_dataloader_common(
        msrvtt_dataset, 
        args, 
        args.batch_size, 
        is_train=True
    )

    return dataloader, len(msrvtt_dataset), train_sampler


def dataloader_msrvtt_test(args, tokenizer, subset="test"):
    """
    Create test/validation dataloader for MSR-VTT dataset.
    
    Args:
        args: Configuration arguments
        tokenizer: Tokenizer for text processing
        subset: Dataset subset to use (test/val)
        
    Returns:
        tuple: (dataloader, dataset_size)
    """
    msrvtt_testset = MSRVTTDataset(
        subset=subset,
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )

    dataloader = _create_dataloader_common(
        msrvtt_testset, 
        args, 
        args.batch_size_val, 
        is_train=False
    )
    
    return dataloader, len(msrvtt_testset)


def dataloader_activity_train(args, tokenizer):
    """
    Create training dataloader for ActivityNet dataset.
    
    Args:
        args: Configuration arguments
        tokenizer: Tokenizer for text processing
        
    Returns:
        tuple: (dataloader, dataset_size, sampler)
    """
    activity_dataset = ActivityNetDataset(
        subset="train",
        data_path=args.anno_path,
        features_path=args.video_path,
        max_words=args.max_words,
        feature_framerate=args.video_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(activity_dataset)
    
    dataloader = _create_dataloader_common(
        activity_dataset, 
        args, 
        args.batch_size, 
        is_train=True
    )

    return dataloader, len(activity_dataset), train_sampler


def dataloader_activity_test(args, tokenizer, subset="test"):
    """
    Create test/validation dataloader for ActivityNet dataset.
    
    Args:
        args: Configuration arguments
        tokenizer: Tokenizer for text processing
        subset: Dataset subset to use (test/val)
        
    Returns:
        tuple: (dataloader, dataset_size)
    """
    activity_testset = ActivityNetDataset(
        subset=subset,
        data_path=args.anno_path,
        features_path=args.video_path,
        max_words=args.max_words,
        feature_framerate=args.video_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames
    )
    
    dataloader = _create_dataloader_common(
        activity_testset, 
        args, 
        args.batch_size_val, 
        is_train=False
    )
    
    return dataloader, len(activity_testset)


def dataloader_didemo_train(args, tokenizer):
    """
    Create training dataloader for DiDeMo dataset.
    
    Args:
        args: Configuration arguments
        tokenizer: Tokenizer for text processing
        
    Returns:
        tuple: (dataloader, dataset_size, sampler)
    """
    didemo_dataset = DiDeMoDataset(
        subset="train",
        data_path=args.anno_path,
        features_path=args.video_path,
        max_words=args.max_words,
        feature_framerate=args.video_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(didemo_dataset)
    
    dataloader = _create_dataloader_common(
        didemo_dataset, 
        args, 
        args.batch_size, 
        is_train=True
    )

    return dataloader, len(didemo_dataset), train_sampler


def dataloader_didemo_test(args, tokenizer, subset="test"):
    """
    Create test/validation dataloader for DiDeMo dataset.
    
    Args:
        args: Configuration arguments
        tokenizer: Tokenizer for text processing
        subset: Dataset subset to use (test/val)
        
    Returns:
        tuple: (dataloader, dataset_size)
    """
    didemo_testset = DiDeMoDataset(
        subset=subset,
        data_path=args.anno_path,
        features_path=args.video_path,
        max_words=args.max_words,
        feature_framerate=args.video_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames
    )
    
    dataloader = _create_dataloader_common(
        didemo_testset, 
        args, 
        args.batch_size_val, 
        is_train=False
    )
    
    return dataloader, len(didemo_testset)


def dataloader_msvd_train(args, tokenizer):
    """
    Create training dataloader for MSVD dataset.
    
    Args:
        args: Configuration arguments
        tokenizer: Tokenizer for text processing
        
    Returns:
        tuple: (dataloader, dataset_size, sampler)
    """
    msvd_dataset = MsvdDataset(
        subset="train",
        data_path=args.anno_path,
        features_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        feature_framerate=args.video_framerate
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)
    
    dataloader = _create_dataloader_common(
        msvd_dataset, 
        args, 
        args.batch_size, 
        is_train=True
    )

    return dataloader, len(msvd_dataset), train_sampler


def dataloader_msvd_test(args, tokenizer, subset="test"):
    """
    Create test/validation dataloader for MSVD dataset.
    
    Args:
        args: Configuration arguments
        tokenizer: Tokenizer for text processing
        subset: Dataset subset to use (test/val)
        
    Returns:
        tuple: (dataloader, dataset_size)
    """
    msvd_testset = MsvdDataset(
        subset=subset,
        data_path=args.anno_path,
        features_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        feature_framerate=args.video_framerate
    )

    # Note: MSVD's original implementation lacked distributed sampler handling
    # and used global batch size unlike other datasets
    dataloader = DataLoader(
        msvd_testset,
        batch_size=args.batch_size_val,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=getattr(args, 'pin_memory', False),
        drop_last=False,
        prefetch_factor=getattr(args, 'prefetch_factor', 2) if args.workers > 0 else None,
        persistent_workers=getattr(args, 'persistent_workers', False) if args.workers > 0 else False,
        timeout=getattr(args, 'timeout', 0)
    )
    
    return dataloader, len(msvd_testset)


# Dictionary mapping dataset names to their respective dataloader functions
DATALOADER_DICT = {
    "msrvtt": {"train": dataloader_msrvtt_train, "val": dataloader_msrvtt_test, "test": None},
    "activity": {"train": dataloader_activity_train, "val": dataloader_activity_test, "test": None},
    "didemo": {"train": dataloader_didemo_train, "val": None, "test": dataloader_didemo_test},
    "msvd": {"train": dataloader_msvd_train, "val": None, "test": dataloader_msvd_test}
}