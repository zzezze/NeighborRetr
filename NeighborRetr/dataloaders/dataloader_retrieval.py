#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base retrieval dataset for NeighborRetr.

This module provides a base dataset class for video-text retrieval
with optimized data loading techniques.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import random
import numpy as np
from torch.utils.data import Dataset
import torch
from torchvision.transforms import (
    Compose, Resize, CenterCrop, ToTensor, Normalize, 
    InterpolationMode, RandomHorizontalFlip, RandomResizedCrop
)
import NeighborRetr.dataloaders.video_transforms as video_transforms


class RetrievalDataset(Dataset):
    """
    General dataset for video-text retrieval with optimized data loading.
    
    This base class implements common functionality for loading and processing
    video and text data across different datasets.
    """

    def __init__(
            self,
            subset,
            anno_path,
            video_path,
            tokenizer,
            max_words=30,
            max_frames=12,
            video_framerate=1,
            image_resolution=224,
            mode='all',
            config=None
    ):
        """
        Initialize the retrieval dataset.
        
        Args:
            subset (str): Dataset split ('train', 'val', 'test')
            anno_path (str): Path to annotations
            video_path (str): Path to video files
            tokenizer: Tokenizer for text processing
            max_words (int): Maximum number of words in text
            max_frames (int): Maximum number of video frames
            video_framerate (int): Frame rate for video sampling
            image_resolution (int): Resolution for image/video frames
            mode (str): Dataset mode ('all', 'text', 'video')
            config: Additional configuration parameters
        """
        self.subset = subset
        self.anno_path = anno_path
        self.video_path = video_path
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_frames = max_frames
        self.video_framerate = video_framerate
        self.image_resolution = image_resolution
        self.mode = mode  # all/text/vision
        self.config = config

        # Load video paths and captions
        self.video_dict, self.sentences_dict = self._get_anns(self.subset)
        self.video_list = list(self.video_dict.keys())
        
        # Initialize video extractor with caching enabled
        self._init_video_extractor()
        
        # Create transforms
        self._init_transforms()
        
        # Special tokens
        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>", 
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]", 
            "UNK_TOKEN": "[UNK]", 
            "PAD_TOKEN": "[PAD]"
        }
        
        # Text tokenization cache
        self.text_cache = {}
        
        # Determine dataset length based on mode
        if self.mode in ['all', 'text']:
            self.sample_len = len(self.sentences_dict)
        else:
            self.sample_len = len(self.video_list)
            
        # Start prefetching if enabled
        if getattr(config, 'use_prefetch', False) and self.mode != 'text':
            self._start_prefetching()

    def _init_video_extractor(self):
        """Initialize the video extractor with caching"""
        from .rawvideo_util import RawVideoExtractor
        
        cache_size = getattr(self.config, 'video_cache_size', 32)
        use_prefetch = getattr(self.config, 'use_prefetch', False)
        
        self.rawVideoExtractor = RawVideoExtractor(
            framerate=self.video_framerate, 
            size=self.image_resolution, 
            cache_size=cache_size,
            use_prefetch=use_prefetch,
            subset=self.subset
        )

    def _init_transforms(self):
        """Initialize image and video transforms"""
        # Standard image normalization parameters for CLIP
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        
        # Default transform for testing
        self.transform = Compose([
            Resize(self.image_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(self.image_resolution),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(mean, std),
        ])
        
        # Create transform dictionary for different use cases
        self.tsfm_dict = {
            'clip_test': Compose([
                Resize(self.image_resolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(self.image_resolution),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(mean, std),
            ]),
            'clip_train': Compose([
                RandomResizedCrop(self.image_resolution, scale=(0.5, 1.0)),
                RandomHorizontalFlip(),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(mean, std),
            ])
        }
        
        # Augmentation transform for training
        self.aug_transform = video_transforms.create_random_augment(
            input_size=(self.image_resolution, self.image_resolution),
            auto_augment='rand-m7-n4-mstd0.5-inc1',
            interpolation='bicubic',
        )

    def _start_prefetching(self):
        """Start background prefetching if supported"""
        if not hasattr(self.rawVideoExtractor, 'start_prefetching'):
            return
            
        if self.mode == 'all':
            # Prefetch the first batch of videos
            batch_size = 16  # Reasonable default batch size for prefetching
            video_paths = []
            
            for i in range(min(batch_size, len(self.sentences_dict))):
                video_id, caption = self.sentences_dict[i]
                _, _, s, e = self._get_text(caption, use_cache=False)
                video_path = self.video_dict[video_id]
                video_paths.append(video_path)
                
            self.rawVideoExtractor.start_prefetching(video_paths)
        elif self.mode == 'video':
            # Prefetch first batch of videos for video-only mode
            batch_size = 16
            video_paths = []
            
            for i in range(min(batch_size, len(self.video_list))):
                video_id = self.video_list[i]
                video_path = self.video_dict[video_id]
                video_paths.append(video_path)
                
            self.rawVideoExtractor.start_prefetching(video_paths)

    def __len__(self):
        """Get the length of the dataset based on mode"""
        return self.sample_len

    def _get_anns(self, subset='train'):
        """
        Get annotations for the dataset.
        
        Args:
            subset: Dataset split to use
            
        Returns:
            tuple: (video_dict, sentences_dict)
                
        Note:
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def _get_text(self, caption, use_cache=True):
        """
        Process and tokenize caption text with caching for efficiency.
        
        Args:
            caption: Caption data (text, start_time, end_time)
            use_cache: Whether to use text cache
            
        Returns:
            tuple: (input_ids, input_mask, start_time, end_time)
        """
        if len(caption) == 3:
            _caption_text, s, e = caption
        else:
            raise ValueError("Caption format must be (text, start_time, end_time)")

        # Handle list of captions by selecting one
        if isinstance(_caption_text, list):
            caption_text = random.choice(_caption_text)
        else:
            caption_text = _caption_text
            
        # Check cache if enabled
        if use_cache and caption_text in self.text_cache:
            return self.text_cache[caption_text]
        
        # Tokenize text
        words = self.tokenizer.tokenize(caption_text)
        
        # Add special tokens
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        # Convert tokens to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)

        # Pad to max length
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words

        # Convert to numpy arrays
        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)
        
        # Cache result
        if use_cache:
            self.text_cache[caption_text] = (input_ids, input_mask, s, e)

        return input_ids, input_mask, s, e

    def _get_rawvideo_dec(self, video_id, s=None, e=None):
        """
        Get video frames using optimized decoder with error handling.
        
        Args:
            video_id: ID of the video to process
            s: Start time in seconds
            e: End time in seconds
            
        Returns:
            tuple: (video, video_mask)
        """
        # Initialize mask and output tensor
        video_mask = np.zeros(self.max_frames, dtype=np.long)
        max_video_length = 0

        # Default empty tensor with correct shape
        video = np.zeros(
            (self.max_frames, 3, self.image_resolution, self.image_resolution), 
            dtype=np.float
        )

        # Get video path
        video_path = self.video_dict[video_id]
        
        try:
            # Get video data using optimized extractor
            video_data = self.rawVideoExtractor.get_video_data(
                video_path, 
                start_time=s, 
                end_time=e
            )
            patch_images = video_data['video']
            
            if isinstance(patch_images, torch.Tensor) and patch_images.ndim > 3:
                # Process frames if we have valid data
                slice_len = min(patch_images.shape[0], self.max_frames)
                max_video_length = slice_len
                
                if slice_len > 0:
                    video[:slice_len] = patch_images[:slice_len].cpu().numpy()
                
                # Set mask for valid frames
                video_mask[:max_video_length] = 1
            else:
                print(f"Warning: Invalid video tensor for {video_id} at {video_path}")
                
        except Exception as e:
            print(f"Error processing video {video_id} at {video_path}: {e}")

        return video, video_mask

    def __getitem__(self, idx):
        """
        Get dataset items based on mode (all/text/video).
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple containing text and/or video data based on mode
        """
        if self.mode == 'all':
            # Return both text and video
            video_id, caption = self.sentences_dict[idx]
            
            # Get text features with caching
            text_ids, text_mask, s, e = self._get_text(caption)
            
            # Get video features with optimized decoder
            video, video_mask = self._get_rawvideo_dec(video_id, s, e)
            
            # Schedule next batch for prefetching if supported
            if idx % 8 == 0 and hasattr(self.rawVideoExtractor, 'start_prefetching'):
                next_idx = min(idx + 16, len(self.sentences_dict) - 1)
                next_video_id, _ = self.sentences_dict[next_idx]
                self.rawVideoExtractor.start_prefetching([self.video_dict[next_video_id]])
            
            return text_ids, text_mask, video, video_mask, idx, hash(video_id.replace("video", ""))
            
        elif self.mode == 'text':
            # Return only text features
            video_id, caption = self.sentences_dict[idx]
            text_ids, text_mask, _, _ = self._get_text(caption)
            return text_ids, text_mask, idx
            
        elif self.mode == 'video':
            # Return only video features
            video_id = self.video_list[idx]
            video, video_mask = self._get_rawvideo_dec(video_id)
            
            # Schedule next batch for prefetching if supported
            if idx % 8 == 0 and hasattr(self.rawVideoExtractor, 'start_prefetching'):
                next_idx = min(idx + 16, len(self.video_list) - 1)
                next_video_id = self.video_list[next_idx]
                self.rawVideoExtractor.start_prefetching([self.video_dict[next_video_id]])
                
            return video, video_mask, idx

    def get_text_len(self):
        """Get number of text samples"""
        return len(self.sentences_dict)

    def get_video_len(self):
        """Get number of video samples"""
        return len(self.video_list)

    def get_text_content(self, ind):
        """Get text content for a given index"""
        return self.sentences_dict[ind][1]

    def get_data_name(self):
        """Get dataset name and subset"""
        return f"{self.__class__.__name__}_{self.subset}"

    def get_vis_info(self, idx):
        """Get visualization info for a sample"""
        video_id, caption = self.sentences_dict[idx]
        video_path = self.video_dict[video_id]
        return caption, video_path


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames.
    
    Args:
        frames: Frames of images sampled from the video
        spatial_idx: Spatial sampling method (-1=random, 0=left/top, 1=center, 2=right/bottom)
        min_scale: Minimum scale for resizing
        max_scale: Maximum scale for resizing
        crop_size: Size for cropping
        random_horizontal_flip: Whether to apply random horizontal flip
        inverse_uniform_sampling: Whether to sample scale inversely
        aspect_ratio: Aspect ratio range for resizing
        scale: Scale range for resizing
        motion_shift: Whether to apply motion shift for resizing
        
    Returns:
        Spatially sampled frames
    """
    assert spatial_idx in [-1, 0, 1, 2]
    
    if spatial_idx == -1:
        # Random spatial sampling for training
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # Deterministic spatial sampling for testing
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
        
    return frames