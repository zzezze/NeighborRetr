#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video transformation utilities for data augmentation in deep learning.

This module provides a comprehensive set of video transformation functions
for data augmentation in computer vision tasks. It includes operations for:
- Spatial transformations (crop, resize, flip)
- Color transformations (jitter, normalization)
- Advanced augmentations (RandAugment integration)

The transformations are designed to work with both individual frames and
batches of video frames, compatible with PyTorch tensor operations.
"""

import math
import numbers
import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms
import PIL
import torchvision

from .rand_augment import rand_augment_transform
from .random_erasing import RandomErasing

# Mapping PIL interpolation methods to string representations
_PIL_INTERPOLATION_TO_STR = {
    Image.NEAREST: "PIL.Image.NEAREST",
    Image.BILINEAR: "PIL.Image.BILINEAR",
    Image.BICUBIC: "PIL.Image.BICUBIC",
    Image.LANCZOS: "PIL.Image.LANCZOS",
    Image.HAMMING: "PIL.Image.HAMMING",
    Image.BOX: "PIL.Image.BOX",
}

# Default interpolation options for random selection
_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def _pil_interp(method):
    """
    Convert interpolation method name to PIL interpolation constant.
    
    Args:
        method (str): Interpolation method name
        
    Returns:
        PIL interpolation constant
    """
    if method == "bicubic":
        return Image.BICUBIC
    elif method == "lanczos":
        return Image.LANCZOS
    elif method == "hamming":
        return Image.HAMMING
    else:
        return Image.BILINEAR


def random_short_side_scale_jitter(
    images, min_size, max_size, boxes=None, inverse_uniform_sampling=False
):
    """
    Perform a spatial short scale jittering on the given images and boxes.
    
    Args:
        images (tensor): Images tensor with shape [num_frames, channel, height, width]
        min_size (int): Minimal size to scale the frames
        max_size (int): Maximal size to scale the frames
        boxes (ndarray, optional): Corresponding boxes with shape [num_boxes, 4]
        inverse_uniform_sampling (bool): If True, sample uniformly in
            [1/max_scale, 1/min_scale] and take reciprocal
            
    Returns:
        tuple: Scaled images tensor and optionally scaled boxes
    """
    if inverse_uniform_sampling:
        size = int(
            round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size))
        )
    else:
        size = int(round(np.random.uniform(min_size, max_size)))

    height = images.shape[2]
    width = images.shape[3]
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return images, boxes
        
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        if boxes is not None:
            boxes = boxes * float(new_height) / height
    else:
        new_width = int(math.floor((float(width) / height) * size))
        if boxes is not None:
            boxes = boxes * float(new_width) / width

    return (
        torch.nn.functional.interpolate(
            images,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        ),
        boxes,
    )


def crop_boxes(boxes, x_offset, y_offset):
    """
    Perform crop on bounding boxes given the offsets.
    
    Args:
        boxes (ndarray): Bounding boxes with shape [num_boxes, 4]
        x_offset (int): Cropping offset in the x axis
        y_offset (int): Cropping offset in the y axis
        
    Returns:
        ndarray: Cropped boxes with shape [num_boxes, 4]
    """
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes


def random_crop(images, size, boxes=None):
    """
    Perform random spatial crop on images and corresponding boxes.
    
    Args:
        images (tensor): Images tensor with shape [num_frames, channel, height, width]
        size (int): Size of height and width to crop
        boxes (ndarray, optional): Corresponding boxes with shape [num_boxes, 4]
        
    Returns:
        tuple: Cropped images tensor and optionally cropped boxes
    """
    if images.shape[2] == size and images.shape[3] == size:
        return images
        
    height = images.shape[2]
    width = images.shape[3]
    y_offset = 0
    if height > size:
        y_offset = int(np.random.randint(0, height - size))
    x_offset = 0
    if width > size:
        x_offset = int(np.random.randint(0, width - size))
    cropped = images[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size
    ]

    cropped_boxes = (
        crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    )

    return cropped, cropped_boxes


def horizontal_flip(prob, images, boxes=None):
    """
    Perform horizontal flip on images and corresponding boxes.
    
    Args:
        prob (float): Probability to flip the images
        images (tensor): Images tensor with shape [num_frames, channel, height, width]
        boxes (ndarray, optional): Corresponding boxes with shape [num_boxes, 4]
        
    Returns:
        tuple: Flipped images tensor and optionally flipped boxes
    """
    if boxes is None:
        flipped_boxes = None
    else:
        flipped_boxes = boxes.copy()

    if np.random.uniform() < prob:
        images = images.flip((-1))

        if len(images.shape) == 3:
            width = images.shape[2]
        elif len(images.shape) == 4:
            width = images.shape[3]
        else:
            raise NotImplementedError("Dimension not supported")
            
        if boxes is not None:
            flipped_boxes[:, [0, 2]] = width - boxes[:, [2, 0]] - 1

    return images, flipped_boxes


def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    """
    Perform uniform spatial sampling on images and corresponding boxes.
    
    Args:
        images (tensor): Images tensor with shape [num_frames, channel, height, width]
        size (int): Size of height and width to crop
        spatial_idx (int): Position index (0, 1, 2) for left/center/right or top/center/bottom
        boxes (ndarray, optional): Corresponding boxes with shape [num_boxes, 4]
        scale_size (int, optional): If provided, resize images to this size before cropping
        
    Returns:
        tuple: Cropped images tensor and optionally cropped boxes
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
            
    cropped = images[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size
    ]
    cropped_boxes = (
        crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    )
    
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped, cropped_boxes


def clip_boxes_to_image(boxes, height, width):
    """
    Clip bounding boxes to image boundaries.
    
    Args:
        boxes (ndarray): Bounding boxes with shape [num_boxes, 4]
        height (int): Image height
        width (int): Image width
        
    Returns:
        ndarray: Clipped boxes with shape [num_boxes, 4]
    """
    clipped_boxes = boxes.copy()
    clipped_boxes[:, [0, 2]] = np.minimum(
        width - 1.0, np.maximum(0.0, boxes[:, [0, 2]])
    )
    clipped_boxes[:, [1, 3]] = np.minimum(
        height - 1.0, np.maximum(0.0, boxes[:, [1, 3]])
    )
    return clipped_boxes


def blend(images1, images2, alpha):
    """
    Blend two images with a given weight alpha.
    
    Args:
        images1 (tensor): First images tensor
        images2 (tensor): Second images tensor
        alpha (float): Blending weight
        
    Returns:
        tensor: Blended images tensor
    """
    return images1 * alpha + images2 * (1 - alpha)


def grayscale(images):
    """
    Convert images to grayscale. The channels should be in BGR order.
    
    Args:
        images (tensor): Images tensor with RGB channels
        
    Returns:
        tensor: Grayscale images tensor
    """
    # R -> 0.299, G -> 0.587, B -> 0.114.
    img_gray = torch.tensor(images)
    gray_channel = (
        0.299 * images[:, 2] + 0.587 * images[:, 1] + 0.114 * images[:, 0]
    )
    img_gray[:, 0] = gray_channel
    img_gray[:, 1] = gray_channel
    img_gray[:, 2] = gray_channel
    return img_gray


def color_jitter(images, img_brightness=0, img_contrast=0, img_saturation=0):
    """
    Perform color jittering on images. The channels should be in BGR order.
    
    Args:
        images (tensor): Images tensor
        img_brightness (float): Jitter ratio for brightness
        img_contrast (float): Jitter ratio for contrast
        img_saturation (float): Jitter ratio for saturation
        
    Returns:
        tensor: Color jittered images tensor
    """
    jitter = []
    if img_brightness != 0:
        jitter.append("brightness")
    if img_contrast != 0:
        jitter.append("contrast")
    if img_saturation != 0:
        jitter.append("saturation")

    if len(jitter) > 0:
        order = np.random.permutation(np.arange(len(jitter)))
        for idx in range(0, len(jitter)):
            if jitter[order[idx]] == "brightness":
                images = brightness_jitter(img_brightness, images)
            elif jitter[order[idx]] == "contrast":
                images = contrast_jitter(img_contrast, images)
            elif jitter[order[idx]] == "saturation":
                images = saturation_jitter(img_saturation, images)
    return images


def brightness_jitter(var, images):
    """
    Perform brightness jittering on images. The channels should be in BGR order.
    
    Args:
        var (float): Jitter ratio for brightness
        images (tensor): Images tensor
        
    Returns:
        tensor: Brightness jittered images tensor
    """
    alpha = 1.0 + np.random.uniform(-var, var)
    img_bright = torch.zeros(images.shape)
    images = blend(images, img_bright, alpha)
    return images


def contrast_jitter(var, images):
    """
    Perform contrast jittering on images. The channels should be in BGR order.
    
    Args:
        var (float): Jitter ratio for contrast
        images (tensor): Images tensor
        
    Returns:
        tensor: Contrast jittered images tensor
    """
    alpha = 1.0 + np.random.uniform(-var, var)
    img_gray = grayscale(images)
    img_gray[:] = torch.mean(img_gray, dim=(1, 2, 3), keepdim=True)
    images = blend(images, img_gray, alpha)
    return images


def saturation_jitter(var, images):
    """
    Perform saturation jittering on images. The channels should be in BGR order.
    
    Args:
        var (float): Jitter ratio for saturation
        images (tensor): Images tensor
        
    Returns:
        tensor: Saturation jittered images tensor
    """
    alpha = 1.0 + np.random.uniform(-var, var)
    img_gray = grayscale(images)
    images = blend(images, img_gray, alpha)
    return images


def lighting_jitter(images, alphastd, eigval, eigvec):
    """
    Perform AlexNet-style PCA jitter on images.
    
    Args:
        images (tensor): Images tensor
        alphastd (float): Standard deviation for alpha
        eigval (list): Eigenvalues for PCA jitter
        eigvec (list): Eigenvectors for PCA jitter
        
    Returns:
        tensor: Lighting jittered images tensor
    """
    if alphastd == 0:
        return images
        
    # Generate alpha1, alpha2, alpha3
    alpha = np.random.normal(0, alphastd, size=(1, 3))
    eig_vec = np.array(eigvec)
    eig_val = np.reshape(eigval, (1, 3))
    rgb = np.sum(
        eig_vec * np.repeat(alpha, 3, axis=0) * np.repeat(eig_val, 3, axis=0),
        axis=1,
    )
    
    out_images = torch.zeros_like(images)
    if len(images.shape) == 3:
        # C H W
        channel_dim = 0
    elif len(images.shape) == 4:
        # T C H W
        channel_dim = 1
    else:
        raise NotImplementedError(f"Unsupported dimension {len(images.shape)}")

    for idx in range(images.shape[channel_dim]):
        # C H W
        if len(images.shape) == 3:
            out_images[idx] = images[idx] + rgb[2 - idx]
        # T C H W
        elif len(images.shape) == 4:
            out_images[:, idx] = images[:, idx] + rgb[2 - idx]
        else:
            raise NotImplementedError(
                f"Unsupported dimension {len(images.shape)}"
            )

    return out_images


def color_normalization(images, mean, stddev):
    """
    Perform color normalization on images.
    
    Args:
        images (tensor): Images tensor
        mean (list): Mean values for each channel
        stddev (list): Standard deviation values for each channel
        
    Returns:
        tensor: Normalized images tensor
    """
    if len(images.shape) == 3:
        assert (
            len(mean) == images.shape[0]
        ), "Channel mean not computed properly"
        assert (
            len(stddev) == images.shape[0]
        ), "Channel stddev not computed properly"
    elif len(images.shape) == 4:
        assert (
            len(mean) == images.shape[1]
        ), "Channel mean not computed properly"
        assert (
            len(stddev) == images.shape[1]
        ), "Channel stddev not computed properly"
    else:
        raise NotImplementedError(f"Unsupported dimension {len(images.shape)}")

    out_images = torch.zeros_like(images)
    for idx in range(len(mean)):
        # C H W
        if len(images.shape) == 3:
            out_images[idx] = (images[idx] - mean[idx]) / stddev[idx]
        # T C H W
        elif len(images.shape) == 4:
            out_images[:, idx] = (images[:, idx] - mean[idx]) / stddev[idx]
        else:
            raise NotImplementedError(
                f"Unsupported dimension {len(images.shape)}"
            )
    return out_images


def _get_param_spatial_crop(
    scale, ratio, height, width, num_repeat=10, log_scale=True, switch_hw=False
):
    """
    Calculate parameters for spatial crop with specified scale and ratio.
    
    Args:
        scale (tuple): Scale range
        ratio (tuple): Aspect ratio range
        height (int): Image height
        width (int): Image width
        num_repeat (int): Number of attempts to find valid crop parameters
        log_scale (bool): Whether to sample aspect ratio in log space
        switch_hw (bool): Whether to randomly switch height and width
        
    Returns:
        tuple: (i, j, h, w) crop parameters
    """
    for _ in range(num_repeat):
        area = height * width
        target_area = random.uniform(*scale) * area
        if log_scale:
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
        else:
            aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if np.random.uniform() < 0.5 and switch_hw:
            w, h = h, w

        if 0 < w <= width and 0 < h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            return i, j, h, w

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


def random_resized_crop(
    images,
    target_height,
    target_width,
    scale=(0.8, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
):
    """
    Crop to random size and aspect ratio, then resize to target size.
    
    Args:
        images (tensor): Images tensor
        target_height (int): Target height
        target_width (int): Target width
        scale (tuple): Scale range
        ratio (tuple): Aspect ratio range
        
    Returns:
        tensor: Cropped and resized images tensor
    """
    height = images.shape[2]
    width = images.shape[3]

    i, j, h, w = _get_param_spatial_crop(scale, ratio, height, width)
    cropped = images[:, :, i : i + h, j : j + w]
    return torch.nn.functional.interpolate(
        cropped,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )


def random_resized_crop_with_shift(
    images,
    target_height,
    target_width,
    scale=(0.8, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
):
    """
    Crops with shift between first and last frame, with interpolated boxes for middle frames.
    
    Args:
        images (tensor): Images tensor
        target_height (int): Target height
        target_width (int): Target width
        scale (tuple): Scale range
        ratio (tuple): Aspect ratio range
        
    Returns:
        tensor: Cropped and resized images tensor with shift
    """
    t = images.shape[1]
    height = images.shape[2]
    width = images.shape[3]

    i, j, h, w = _get_param_spatial_crop(scale, ratio, height, width)
    i_, j_, h_, w_ = _get_param_spatial_crop(scale, ratio, height, width)
    i_s = [int(i) for i in torch.linspace(i, i_, steps=t).tolist()]
    j_s = [int(i) for i in torch.linspace(j, j_, steps=t).tolist()]
    h_s = [int(i) for i in torch.linspace(h, h_, steps=t).tolist()]
    w_s = [int(i) for i in torch.linspace(w, w_, steps=t).tolist()]
    out = torch.zeros((3, t, target_height, target_width))
    for ind in range(t):
        out[:, ind : ind + 1, :, :] = torch.nn.functional.interpolate(
            images[
                :,
                ind : ind + 1,
                i_s[ind] : i_s[ind] + h_s[ind],
                j_s[ind] : j_s[ind] + w_s[ind],
            ],
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
    return out


def create_random_augment(
    input_size,
    auto_augment=None,
    interpolation="bilinear",
):
    """
    Create a random augmentation transform.
    
    Args:
        input_size (tuple or int): Size of input images
        auto_augment (str): RandAugment parameters, e.g., "rand-m7-n4-mstd0.5-inc1"
        interpolation (str): Interpolation method
        
    Returns:
        transforms.Compose: Composed augmentation transforms
    """
    if isinstance(input_size, tuple):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = {"translate_const": int(img_size_min * 0.45)}
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = _pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            return transforms.Compose(
                [rand_augment_transform(auto_augment, aa_params)]
            )
    raise NotImplementedError


def random_sized_crop_img(
    im,
    size,
    jitter_scale=(0.08, 1.0),
    jitter_aspect=(3.0 / 4.0, 4.0 / 3.0),
    max_iter=10,
):
    """
    Perform Inception-style random sized crop on a single image.
    
    Args:
        im (tensor): Image tensor [C, H, W]
        size (int): Target size
        jitter_scale (tuple): Scale range
        jitter_aspect (tuple): Aspect ratio range
        max_iter (int): Maximum iterations to find valid crop
        
    Returns:
        tensor: Cropped and resized image tensor
    """
    assert (
        len(im.shape) == 3
    ), "Currently only support image for random_sized_crop"
    h, w = im.shape[1:3]
    i, j, h, w = _get_param_spatial_crop(
        scale=jitter_scale,
        ratio=jitter_aspect,
        height=h,
        width=w,
        num_repeat=max_iter,
        log_scale=False,
        switch_hw=True,
    )
    cropped = im[:, i : i + h, j : j + w]
    return torch.nn.functional.interpolate(
        cropped.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


class RandomResizedCropAndInterpolation:
    """
    Crop to random size and aspect ratio, then resize with random interpolation.
    
    Args:
        size (int or tuple): Target size
        scale (tuple): Scale range, default: (0.08, 1.0)
        ratio (tuple): Aspect ratio range, default: (3/4, 4/3)
        interpolation (str or tuple): Interpolation method(s)
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation="bilinear",
    ):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            print("Range should be of kind (min, max)")

        if interpolation == "random":
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """
        Get parameters for random sized crop.
        
        Args:
            img (PIL.Image): Input image
            scale (tuple): Scale range
            ratio (tuple): Aspect ratio range
            
        Returns:
            tuple: (i, j, h, w) crop parameters
        """
        area = img.size[0] * img.size[1]

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped and resized
            
        Returns:
            PIL.Image: Randomly cropped and resized image
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return F.resized_crop(img, i, j, h, w, self.size, interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = " ".join(
                [_PIL_INTERPOLATION_TO_STR[x] for x in self.interpolation]
            )
        else:
            interpolate_str = _PIL_INTERPOLATION_TO_STR[self.interpolation]
        format_string = self.__class__.__name__ + "(size={0}".format(self.size)
        format_string += ", scale={0}".format(
            tuple(round(s, 4) for s in self.scale)
        )
        format_string += ", ratio={0}".format(
            tuple(round(r, 4) for r in self.ratio)
        )
        format_string += ", interpolation={0})".format(interpolate_str)
        return format_string


def transforms_imagenet_train(
    img_size=224,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment=None,
    interpolation="random",
    use_prefetcher=False,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    re_prob=0.0,
    re_mode="const",
    re_count=1,
    re_num_splits=0,
    separate=False,
):
    """
    Create training transform pipeline for ImageNet-style datasets.
    
    Args:
        img_size (int or tuple): Image size
        scale (tuple): Scale range for RandomResizedCrop
        ratio (tuple): Aspect ratio range for RandomResizedCrop
        hflip (float): Probability of horizontal flip
        vflip (float): Probability of vertical flip
        color_jitter (float or tuple): Color jitter parameters
        auto_augment (str): Auto augment policy
        interpolation (str): Interpolation method
        use_prefetcher (bool): Whether using prefetcher
        mean (tuple): Mean for normalization
        std (tuple): Standard deviation for normalization
        re_prob (float): Random erasing probability
        re_mode (str): Random erasing mode
        re_count (int): Random erasing count
        re_num_splits (int): Number of splits for random erasing
        separate (bool): If True, returns transforms as separate components
        
    Returns:
        transforms.Compose or tuple: Training transforms
    """
    if isinstance(img_size, tuple):
        img_size = img_size[-2:]
    else:
        img_size = img_size

    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
    
    # Primary transforms
    primary_tfl = [
        RandomResizedCropAndInterpolation(
            img_size, scale=scale, ratio=ratio, interpolation=interpolation
        )
    ]
    
    if hflip > 0.0:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.0:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    # Secondary transforms
    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = _pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith("augmix"):
            raise NotImplementedError("Augmix not implemented")
        else:
            raise NotImplementedError("Auto aug not implemented")
    elif color_jitter is not None:
        # color jitter is enabled when not using AA
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    # Final transforms
    final_tfl = []
    final_tfl += [
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ]
    if re_prob > 0.0:
        final_tfl.append(
            RandomErasing(
                re_prob,
                mode=re_mode,
                max_count=re_count,
                num_splits=re_num_splits,
                device="cpu",
                cube=False,
            )
        )

    if separate:
        return (
            transforms.Compose(primary_tfl),
            transforms.Compose(secondary_tfl),
            transforms.Compose(final_tfl),
        )
    else:
        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)


class Compose:
    """
    Compose several video transforms together.
    
    Similar to torchvision.transforms.Compose but for video clip transforms.
    
    Args:
        transforms (list): List of transforms to compose
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        """
        Args:
            clip (list): List of PIL.Images or numpy.ndarrays
            
        Returns:
            list: Transformed clip
        """
        for transform in self.transforms:
            clip = transform(clip)
        return clip


class RandomHorizontalFlip:
    """
    Horizontally flip a list of images randomly with probability 0.5.
    """

    def __call__(self, clip):
        """
        Args:
            clip (list): List of PIL.Images or numpy.ndarrays
            
        Returns:
            list: Randomly flipped clip
        """
        if random.random() < 0.5:
            if isinstance(clip[0], np.ndarray):
                return [np.fliplr(img) for img in clip]
            elif isinstance(clip[0], PIL.Image.Image):
                return [img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip]
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                              ' but got list of {0}'.format(type(clip[0])))
        return clip


class RandomResize:
    """
    Resize video frames by a random scaling factor.
    
    Args:
        ratio (tuple): Range of scaling ratios (min, max)
        interpolation (str): Interpolation method
    """

    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        """
        Args:
            clip (list): List of PIL.Images or numpy.ndarrays
            
        Returns:
            list: Resized clip
        """
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        
        import torchvision.transforms.functional as F
        resized = F.resize_clip(
            clip, new_size, interpolation=self.interpolation)
        return resized


class Resize:
    """
    Resize video frames to a specified size.
    
    Args:
        size (tuple): Target size (width, height)
        interpolation (str): Interpolation method
    """

    def __init__(self, size, interpolation='nearest'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        """
        Args:
            clip (list): List of PIL.Images or numpy.ndarrays
            
        Returns:
            list: Resized clip
        """
        import torchvision.transforms.functional as F
        resized = F.resize_clip(
            clip, self.size, interpolation=self.interpolation)
        return resized


class RandomCrop:
    """
    Crop video frames at a random position.
    
    Args:
        size (int or tuple): Desired output size
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)
        self.size = size

    def __call__(self, clip):
        """
        Args:
            clip (list): List of PIL.Images or numpy.ndarrays
            
        Returns:
            list: Cropped clip
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                          'but got list of {0}'.format(type(clip[0])))
                          
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger than '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)
        
        import torchvision.transforms.functional as F
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return cropped


class ThreeCrop:
    """
    Crop video frames at three positions (left/center/right or top/center/bottom).
    
    Args:
        size (int or tuple): Desired output size
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)
        self.size = size

    def __call__(self, clip):
        """
        Args:
            clip (list): List of PIL.Images or numpy.ndarrays
            
        Returns:
            list: Concatenated list of cropped clips
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                          'but got list of {0}'.format(type(clip[0])))
                          
        if w != im_w and h != im_h:
            import torchvision.transforms.functional as F
            clip = F.resize_clip(clip, self.size, interpolation="bilinear")
            im_h, im_w, im_c = clip[0].shape

        step = np.max((np.max((im_w, im_h)) - self.size[0]) // 2, 0)
        cropped = []
        
        import torchvision.transforms.functional as F
        for i in range(3):
            if (im_h > self.size[0]):
                x1 = 0
                y1 = i * step
                cropped.extend(F.crop_clip(clip, y1, x1, h, w))
            else:
                x1 = i * step
                y1 = 0
                cropped.extend(F.crop_clip(clip, y1, x1, h, w))
        return cropped


class RandomRotation:
    """
    Rotate video frames by a random angle.
    
    Args:
        degrees (int or tuple): Range of degrees to select from
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                              'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                              'it must be of len 2.')
        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
            clip (list): List of PIL.Images or numpy.ndarrays
            
        Returns:
            list: Rotated clip
        """
        import skimage
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [skimage.transform.rotate(img, angle) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                          'but got list of {0}'.format(type(clip[0])))
        return rotated


class CenterCrop:
    """
    Crop video frames at the center.
    
    Args:
        size (int or tuple): Desired output size
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)
        self.size = size

    def __call__(self, clip):
        """
        Args:
            clip (list): List of PIL.Images or numpy.ndarrays
            
        Returns:
            list: Center cropped clip
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                          'but got list of {0}'.format(type(clip[0])))
                          
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger than '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.))
        y1 = int(round((im_h - h) / 2.))
        
        import torchvision.transforms.functional as F
        cropped = F.crop_clip(clip, y1, x1, h, w)

        return cropped


class ColorJitter:
    """
    Randomly change brightness, contrast, saturation, and hue of video frames.
    
    Args:
        brightness (float): Brightness jitter factor
        contrast (float): Contrast jitter factor
        saturation (float): Saturation jitter factor
        hue (float): Hue jitter factor
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        """
        Get jittering parameters.
        
        Returns:
            tuple: Jittering factors
        """
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
            
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
            clip (list): List of PIL.Images
            
        Returns:
            list: Color jittered clip
        """
        if isinstance(clip[0], np.ndarray):
            raise TypeError(
                'Color jitter not yet implemented for numpy arrays')
        elif isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all images
            jittered_clip = []
            for img in clip:
                for func in img_transforms:
                    jittered_img = func(img)
                jittered_clip.append(jittered_img)

        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                          'but got list of {0}'.format(type(clip[0])))
        return jittered_clip


class Normalize:
    """
    Normalize video frames with mean and standard deviation.
    
    Args:
        mean (sequence): Sequence of means for each channel
        std (sequence): Sequence of standard deviations for each channel
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip):
        """
        Args:
            clip (tensor): Video clip tensor
            
        Returns:
            tensor: Normalized video clip tensor
        """
        import torchvision.transforms.functional as F
        return F.normalize(clip, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)