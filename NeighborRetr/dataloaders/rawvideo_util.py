#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video frame extraction utilities optimized for deep learning pipelines.

This module provides efficient video frame extraction capabilities with features:
- Frame caching with LRU policy for memory efficiency
- Optimized video decoding with batch processing
- Multi-threaded prefetching option for improved loading performance
"""

import torch as th
import numpy as np
from PIL import Image
import cv2
import os
import threading
import queue
import time
from torchvision.transforms import (
    Compose, Resize, CenterCrop, ToTensor, Normalize,
    InterpolationMode, RandomResizedCrop, RandomHorizontalFlip
)
import NeighborRetr.dataloaders.video_transforms as video_transforms
from .random_erasing import RandomErasing


class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache implementation.
    
    Provides efficient caching of video frames with automatic eviction
    of least recently accessed items when capacity is reached.
    
    Args:
        capacity (int): Maximum number of items to store in cache
    """
    def __init__(self, capacity=32):
        self.cache = {}
        self.capacity = capacity
        self.timestamps = {}
        self._lock = threading.Lock()

    def get(self, key):
        """
        Retrieve an item from the cache.
        
        Args:
            key: Cache key to lookup
            
        Returns:
            Value associated with key or None if not found
        """
        with self._lock:
            if key not in self.cache:
                return None
            # Update timestamp
            self.timestamps[key] = time.time()
            return self.cache[key]

    def put(self, key, value):
        """
        Add or update an item in the cache.
        
        Args:
            key: Cache key
            value: Value to store
        """
        with self._lock:
            if self.capacity <= 0:
                return

            # If key already exists, just update the timestamp
            if key in self.cache:
                self.cache[key] = value
                self.timestamps[key] = time.time()
                return

            # If cache is full, remove the least recently used item
            if len(self.cache) >= self.capacity:
                oldest_key = min(self.timestamps, key=self.timestamps.get)
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]

            # Add new item
            self.cache[key] = value
            self.timestamps[key] = time.time()


class RawVideoExtractorCV2:
    """
    Optimized video frame extractor using OpenCV.
    
    Features:
    - Frame caching with LRU policy
    - Optimized video decoding with batch processing
    - Memory-efficient frame handling
    - Multi-threaded prefetching option
    
    Args:
        centercrop (bool): Whether to apply center cropping
        size (int): Output frame size (height and width)
        framerate (int): Target framerate for extraction, -1 for original
        subset (str): Dataset subset ("train" or "test")
        cache_size (int): Size of frame cache
        use_prefetch (bool): Whether to use background prefetching
    """
    def __init__(self, centercrop=False, size=224, framerate=-1, 
                 subset="test", cache_size=16, use_prefetch=False):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(self.size)
        self.subset = subset
        self.use_prefetch = use_prefetch
        
        # Initialize frame cache 
        self.frame_cache = LRUCache(capacity=cache_size)
        
        # Transforms dictionary
        self.tsfm_dict = {
            'clip_test': Compose([
                Resize(size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(size),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), 
                          (0.26862954, 0.26130258, 0.27577711)),
            ]),
            'clip_train': Compose([
                RandomResizedCrop(size, scale=(0.5, 1.0)),
                RandomHorizontalFlip(),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), 
                          (0.26862954, 0.26130258, 0.27577711)),
            ])
        }
        
        # Augmentation transform
        self.aug_transform = video_transforms.create_random_augment(
            input_size=(size, size),
            auto_augment='rand-m7-n4-mstd0.5-inc1',
            interpolation='bicubic',
        )
        
        # For multi-threaded prefetching
        if self.use_prefetch:
            self.prefetch_queue = queue.Queue(maxsize=8)
            self.prefetch_thread = None
            self.prefetch_stopped = threading.Event()
    
    def _transform(self, n_px):
        """
        Create the default transform pipeline.
        
        Args:
            n_px (int): Size for resizing and cropping
            
        Returns:
            torchvision.transforms.Compose: Transform pipeline
        """
        return Compose([
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), 
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _generate_frame_indices(self, fps, total_frames, sample_fp, start_sec, end_sec):
        """
        Generate optimized frame indices to extract from video.
        
        Args:
            fps (int): Frames per second of video
            total_frames (int): Total number of frames in video
            sample_fp (int): Frames to sample per second
            start_sec (int): Start time in seconds
            end_sec (int): End time in seconds
            
        Returns:
            list: Frame indices to extract
        """
        if sample_fp <= 0:
            sample_fp = fps
            
        interval = max(1, fps // sample_fp)
        
        # Generate frame indices for each second
        frame_indices = []
        for sec in range(start_sec, end_sec + 1):
            sec_base = int(sec * fps)
            for idx in range(0, fps, interval)[:sample_fp]:
                frame_idx = sec_base + idx
                if frame_idx < total_frames:
                    frame_indices.append(frame_idx)
                    
        return frame_indices
    
    def _get_cached_video_key(self, video_path, start_time, end_time, sample_fp):
        """
        Generate a unique key for caching based on video path and parameters.
        
        Args:
            video_path (str): Path to video file
            start_time (int): Start time in seconds
            end_time (int): End time in seconds
            sample_fp (int): Frames to sample per second
            
        Returns:
            str: Cache key
        """
        file_mtime = os.path.getmtime(video_path) if os.path.exists(video_path) else 0
        return f"{video_path}_{file_mtime}_{start_time}_{end_time}_{sample_fp}"
    
    def video_to_tensor(self, video_file, preprocess, sample_fp=0, 
                        start_time=None, end_time=None, _no_process=False):
        """
        Extract frames from video and convert to tensor with optimized performance.
        
        Args:
            video_file (str): Path to the video file
            preprocess (callable): Transform to apply to each frame
            sample_fp (int): Number of frames to sample per second
            start_time (int, optional): Start time in seconds
            end_time (int, optional): End time in seconds
            _no_process (bool): If True, return PIL images without processing
            
        Returns:
            dict: Dictionary containing video tensor
        """
        if not os.path.exists(video_file):
            print(f"Warning: Video file not found: {video_file}")
            return {'video': th.zeros((1, 3, self.size, self.size))}
            
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
                   
        # Check cache first
        cache_key = self._get_cached_video_key(video_file, start_time, end_time, sample_fp)
        cached_data = self.frame_cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Open video file
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_file}")
            return {'video': th.zeros((1, 3, self.size, self.size))}
            
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if fps == 0:
            print(f"Warning: Invalid FPS (0) for video: {video_file}")
            fps = 30  # Default fallback
            
        # Set time boundaries
        total_duration = max(1, (frame_count + fps - 1) // fps)
        start_sec = 0 if start_time is None else min(start_time, total_duration - 1)
        end_sec = total_duration - 1 if end_time is None else min(end_time, total_duration - 1)
        
        # Generate frame indices to extract
        frame_indices = self._generate_frame_indices(
            fps, frame_count, sample_fp, start_sec, end_sec)
        
        # Read frames efficiently
        images = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(frame_rgb).convert("RGB"))
        
        cap.release()
        
        # Process frames
        if len(images) > 0:
            if _no_process:
                video_data = images
            else:
                if self.subset == "train":
                    # Apply augmentation for training
                    images = self.aug_transform(images)
                
                # Stack processed frames
                try:
                    video_data = th.stack([preprocess(img) for img in images])
                except RuntimeError as e:
                    print(f"Error processing frames from {video_file}: {e}")
                    return {'video': th.zeros((1, 3, self.size, self.size))}
        else:
            video_data = th.zeros((1, 3, self.size, self.size))
        
        # Cache result
        result = {'video': video_data}
        self.frame_cache.put(cache_key, result)
        
        return result

    def get_video_data(self, video_path, start_time=None, end_time=None, _no_process=False):
        """
        Get processed video data from file path.
        
        Args:
            video_path (str): Path to video file
            start_time (int, optional): Start time in seconds
            end_time (int, optional): End time in seconds
            _no_process (bool): If True, return PIL images without processing
            
        Returns:
            dict: Dictionary containing video tensor
        """
        return self.video_to_tensor(
            video_path, 
            self.transform, 
            sample_fp=self.framerate, 
            start_time=start_time,
            end_time=end_time, 
            _no_process=_no_process
        )

    def process_raw_data(self, raw_video_data):
        """
        Reshape raw video tensor for processing.
        
        Args:
            raw_video_data (torch.Tensor): Video tensor
            
        Returns:
            torch.Tensor: Reshaped tensor
        """
        if not isinstance(raw_video_data, th.Tensor):
            return th.zeros((1, 1, 3, self.size, self.size))
            
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        """
        Reorder frames based on specified order.
        
        Args:
            raw_video_data (torch.Tensor): Video tensor
            frame_order (int): Order type (0: original, 1: reverse, 2: random)
            
        Returns:
            torch.Tensor: Reordered video tensor
        """
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0 or not isinstance(raw_video_data, th.Tensor):
            return raw_video_data
            
        if frame_order == 1:
            # Reverse order
            return raw_video_data.flip(0)
        elif frame_order == 2:
            # Random order
            idx = th.randperm(raw_video_data.size(0))
            return raw_video_data[idx]
        
        return raw_video_data

    def start_prefetching(self, video_paths, start_times=None, end_times=None):
        """
        Start a background thread to prefetch video frames.
        
        Args:
            video_paths (list): List of video file paths
            start_times (list, optional): List of start times
            end_times (list, optional): List of end times
        """
        if not self.use_prefetch:
            return
            
        # Stop existing thread if any
        self.stop_prefetching()
        
        # Start new prefetch thread
        self.prefetch_stopped.clear()
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(video_paths, start_times, end_times),
            daemon=True
        )
        self.prefetch_thread.start()
    
    def _prefetch_worker(self, video_paths, start_times=None, end_times=None):
        """
        Worker function to prefetch videos in background.
        
        Args:
            video_paths (list): List of video file paths
            start_times (list, optional): List of start times
            end_times (list, optional): List of end times
        """
        if start_times is None:
            start_times = [None] * len(video_paths)
        if end_times is None:
            end_times = [None] * len(video_paths)
            
        for path, start, end in zip(video_paths, start_times, end_times):
            if self.prefetch_stopped.is_set():
                break
                
            # Get cache key
            cache_key = self._get_cached_video_key(path, start, end, self.framerate)
            
            # Skip if already cached
            if self.frame_cache.get(cache_key) is not None:
                continue
                
            # Load video and add to cache
            result = self.get_video_data(path, start, end)
            
            # Try to put in queue for the dataset to consume
            try:
                self.prefetch_queue.put((path, result), block=False)
            except queue.Full:
                pass
    
    def stop_prefetching(self):
        """
        Stop the prefetching thread.
        """
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            self.prefetch_stopped.set()
            self.prefetch_thread.join(timeout=1.0)
            
            # Clear queue
            while not self.prefetch_queue.empty():
                try:
                    self.prefetch_queue.get(block=False)
                except queue.Empty:
                    break
            
            self.prefetch_thread = None


# Alias for backward compatibility
RawVideoExtractor = RawVideoExtractorCV2