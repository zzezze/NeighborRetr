#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RandomErasing: Image data augmentation technique implementation.

This module implements RandomErasing as described in the paper:
'Random Erasing Data Augmentation' by Zhong et al.
See https://arxiv.org/pdf/1708.04896.pdf

This implementation is designed to be applied to a batch or single 
image tensor after normalization by dataset mean and std.

Based on the implementation from pytorch-image-models:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/random_erasing.py
published under Apache License 2.0.
"""

import math
import random
import torch


def _get_pixels(
    per_pixel, rand_color, patch_size, dtype=torch.float32, device="cuda"
):
    """
    Generate pixel values for the erasing patch.
    
    Args:
        per_pixel (bool): Whether to use per-pixel random values
        rand_color (bool): Whether to use per-channel random values
        patch_size (tuple): Size of the patch (channels, height, width)
        dtype (torch.dtype): Data type for tensor
        device (str): Device for tensor creation
        
    Returns:
        torch.Tensor: Tensor with random pixel values for erasing
    """
    # NOTE: CUDA illegal memory access errors can be caused by the normal_() paths
    # Issue has been fixed in PyTorch master: https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty(
            (patch_size[0], 1, 1), dtype=dtype, device=device
        ).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomErasing:
    """
    Randomly selects a rectangle region in an image and erases its pixels.
    
    This augmentation helps models become more robust to occlusion and improves
    generalization, particularly useful for tasks like person re-identification.
    
    Args:
        probability (float): Probability of erasing operation being applied
        min_area (float): Minimum percentage of erased area relative to image
        max_area (float): Maximum percentage of erased area relative to image
        min_aspect (float): Minimum aspect ratio of erased area
        max_aspect (float, optional): Maximum aspect ratio of erased area
        mode (str): Pixel color mode ('const', 'rand', or 'pixel')
        min_count (int): Minimum number of erasing blocks per image
        max_count (int, optional): Maximum number of erasing blocks per image
        num_splits (int): Number of image splits (for augmix compatibility)
        device (str): Device for tensor operations
        cube (bool): Whether to apply same erasing to entire batch
    """

    def __init__(
        self,
        probability=0.5,
        min_area=0.02,
        max_area=1/3,
        min_aspect=0.3,
        max_aspect=None,
        mode="const",
        min_count=1,
        max_count=None,
        num_splits=0,
        device="cuda",
        cube=True,
    ):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        
        # Calculate aspect ratio bounds in log space
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        
        # Erasing count parameters
        self.min_count = min_count
        self.max_count = max_count or min_count
        
        # For compatibility with augmix
        self.num_splits = num_splits
        
        # Set up pixel generation mode
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        self.cube = cube
        
        if mode == "rand":
            self.rand_color = True  # per block random normal
        elif mode == "pixel":
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == "const"
            
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        """
        Apply erasing to a single image.
        
        Args:
            img (torch.Tensor): Image tensor to erase
            chan (int): Number of channels
            img_h (int): Image height
            img_w (int): Image width
            dtype (torch.dtype): Data type of tensor
        """
        if random.random() > self.probability:
            return
            
        area = img_h * img_w
        count = (
            self.min_count
            if self.min_count == self.max_count
            else random.randint(self.min_count, self.max_count)
        )
        
        for _ in range(count):
            for _ in range(10):  # Try 10 times to find valid parameters
                target_area = (
                    random.uniform(self.min_area, self.max_area) * area / count
                )
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top : top + h, left : left + w] = _get_pixels(
                        self.per_pixel,
                        self.rand_color,
                        (chan, h, w),
                        dtype=dtype,
                        device=self.device,
                    )
                    break

    def _erase_cube(
        self,
        img,
        batch_start,
        batch_size,
        chan,
        img_h,
        img_w,
        dtype,
    ):
        """
        Apply identical erasing to all images in a batch (cube mode).
        
        Args:
            img (torch.Tensor): Batch of images to erase
            batch_start (int): Starting index in batch
            batch_size (int): Size of batch
            chan (int): Number of channels
            img_h (int): Image height
            img_w (int): Image width
            dtype (torch.dtype): Data type of tensor
        """
        if random.random() > self.probability:
            return
            
        area = img_h * img_w
        count = (
            self.min_count
            if self.min_count == self.max_count
            else random.randint(self.min_count, self.max_count)
        )
        
        for _ in range(count):
            for _ in range(100):  # Try more times for batch mode
                target_area = (
                    random.uniform(self.min_area, self.max_area) * area / count
                )
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    for i in range(batch_start, batch_size):
                        img_instance = img[i]
                        img_instance[
                            :, top : top + h, left : left + w
                        ] = _get_pixels(
                            self.per_pixel,
                            self.rand_color,
                            (chan, h, w),
                            dtype=dtype,
                            device=self.device,
                        )
                    break

    def __call__(self, input):
        """
        Apply random erasing to input tensor.
        
        Args:
            input (torch.Tensor): Image tensor or batch of image tensors
            
        Returns:
            torch.Tensor: Tensor with random areas erased
        """
        if len(input.size()) == 3:
            # Single image case
            self._erase(input, *input.size(), input.dtype)
        else:
            # Batch of images case
            batch_size, chan, img_h, img_w = input.size()
            # Skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = (
                batch_size // self.num_splits if self.num_splits > 1 else 0
            )
            
            if self.cube:
                # Apply same erasing pattern to all images in batch
                self._erase_cube(
                    input,
                    batch_start,
                    batch_size,
                    chan,
                    img_h,
                    img_w,
                    input.dtype,
                )
            else:
                # Apply different erasing pattern to each image
                for i in range(batch_start, batch_size):
                    self._erase(input[i], chan, img_h, img_w, input.dtype)
                    
        return input