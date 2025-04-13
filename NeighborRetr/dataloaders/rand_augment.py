#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RandAugment: Practical automated data augmentation implementation.

This module implements RandAugment as described in the paper:
'RandAugment: Practical automated data augmentation with a reduced search space'
https://arxiv.org/abs/1909.13719

Based on the implementation from pytorch-image-models:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/auto_augment.py
published under Apache License 2.0.
"""

import math
import numpy as np
import random
import re
import PIL
from PIL import Image, ImageEnhance, ImageOps

# Constants
_PIL_VER = tuple([int(x) for x in PIL.__version__.split(".")[:2]])
_FILL = (128, 128, 128)
_MAX_LEVEL = 10.0
_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)
_HPARAMS_DEFAULT = {
    "translate_const": 250,
    "img_mean": _FILL,
}


def _interpolation(kwargs):
    """
    Helper function to select interpolation method.
    
    Args:
        kwargs (dict): Keyword arguments that may contain 'resample' parameter
        
    Returns:
        PIL.Image interpolation method
    """
    interpolation = kwargs.pop("resample", Image.BILINEAR)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    else:
        return interpolation


def _check_args_tf(kwargs):
    """
    Ensure compatibility of arguments with different PIL versions.
    
    Args:
        kwargs (dict): Keyword arguments for image transformation
    """
    if "fillcolor" in kwargs and _PIL_VER < (5, 0):
        kwargs.pop("fillcolor")
    kwargs["resample"] = _interpolation(kwargs)


def shear_x(img, factor, **kwargs):
    """
    Apply horizontal shear transformation.
    
    Args:
        img (PIL.Image): Image to transform
        factor (float): Shear factor
        **kwargs: Additional arguments for transformation
        
    Returns:
        PIL.Image: Transformed image
    """
    _check_args_tf(kwargs)
    return img.transform(
        img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs
    )


def shear_y(img, factor, **kwargs):
    """
    Apply vertical shear transformation.
    
    Args:
        img (PIL.Image): Image to transform
        factor (float): Shear factor
        **kwargs: Additional arguments for transformation
        
    Returns:
        PIL.Image: Transformed image
    """
    _check_args_tf(kwargs)
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs
    )


def translate_x_rel(img, pct, **kwargs):
    """
    Translate image horizontally by relative percentage.
    
    Args:
        img (PIL.Image): Image to transform
        pct (float): Translation percentage relative to image width
        **kwargs: Additional arguments for transformation
        
    Returns:
        PIL.Image: Transformed image
    """
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(
        img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs
    )


def translate_y_rel(img, pct, **kwargs):
    """
    Translate image vertically by relative percentage.
    
    Args:
        img (PIL.Image): Image to transform
        pct (float): Translation percentage relative to image height
        **kwargs: Additional arguments for transformation
        
    Returns:
        PIL.Image: Transformed image
    """
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs
    )


def translate_x_abs(img, pixels, **kwargs):
    """
    Translate image horizontally by absolute pixels.
    
    Args:
        img (PIL.Image): Image to transform
        pixels (int): Translation amount in pixels
        **kwargs: Additional arguments for transformation
        
    Returns:
        PIL.Image: Transformed image
    """
    _check_args_tf(kwargs)
    return img.transform(
        img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs
    )


def translate_y_abs(img, pixels, **kwargs):
    """
    Translate image vertically by absolute pixels.
    
    Args:
        img (PIL.Image): Image to transform
        pixels (int): Translation amount in pixels
        **kwargs: Additional arguments for transformation
        
    Returns:
        PIL.Image: Transformed image
    """
    _check_args_tf(kwargs)
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs
    )


def rotate(img, degrees, **kwargs):
    """
    Rotate image by specified degrees.
    
    Args:
        img (PIL.Image): Image to transform
        degrees (float): Rotation angle in degrees
        **kwargs: Additional arguments for transformation
        
    Returns:
        PIL.Image: Transformed image
    """
    _check_args_tf(kwargs)
    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, **kwargs)
    elif _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(
            -rotn_center[0] - post_trans[0],
            -rotn_center[1] - post_trans[1],
            matrix,
        )
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, **kwargs)
    else:
        return img.rotate(degrees, resample=kwargs["resample"])


def auto_contrast(img, **__):
    """
    Apply auto contrast to the image.
    
    Args:
        img (PIL.Image): Image to transform
        
    Returns:
        PIL.Image: Transformed image
    """
    return ImageOps.autocontrast(img)


def invert(img, **__):
    """
    Invert the image colors.
    
    Args:
        img (PIL.Image): Image to transform
        
    Returns:
        PIL.Image: Transformed image
    """
    return ImageOps.invert(img)


def equalize(img, **__):
    """
    Equalize the image histogram.
    
    Args:
        img (PIL.Image): Image to transform
        
    Returns:
        PIL.Image: Transformed image
    """
    return ImageOps.equalize(img)


def solarize(img, thresh, **__):
    """
    Apply solarize effect with threshold.
    
    Args:
        img (PIL.Image): Image to transform
        thresh (int): Threshold value for solarization
        
    Returns:
        PIL.Image: Transformed image
    """
    return ImageOps.solarize(img, thresh)


def solarize_add(img, add, thresh=128, **__):
    """
    Enhanced solarize with add parameter.
    
    Args:
        img (PIL.Image): Image to transform
        add (int): Value to add to pixels below threshold
        thresh (int, optional): Threshold value. Defaults to 128.
        
    Returns:
        PIL.Image: Transformed image
    """
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        return img


def posterize(img, bits_to_keep, **__):
    """
    Reduce image color depth (posterize).
    
    Args:
        img (PIL.Image): Image to transform
        bits_to_keep (int): Number of bits to keep for posterization
        
    Returns:
        PIL.Image: Transformed image
    """
    if bits_to_keep >= 8:
        return img
    return ImageOps.posterize(img, bits_to_keep)


def contrast(img, factor, **__):
    """
    Adjust image contrast.
    
    Args:
        img (PIL.Image): Image to transform
        factor (float): Contrast adjustment factor
        
    Returns:
        PIL.Image: Transformed image
    """
    return ImageEnhance.Contrast(img).enhance(factor)


def color(img, factor, **__):
    """
    Adjust image color saturation.
    
    Args:
        img (PIL.Image): Image to transform
        factor (float): Color adjustment factor
        
    Returns:
        PIL.Image: Transformed image
    """
    return ImageEnhance.Color(img).enhance(factor)


def brightness(img, factor, **__):
    """
    Adjust image brightness.
    
    Args:
        img (PIL.Image): Image to transform
        factor (float): Brightness adjustment factor
        
    Returns:
        PIL.Image: Transformed image
    """
    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img, factor, **__):
    """
    Adjust image sharpness.
    
    Args:
        img (PIL.Image): Image to transform
        factor (float): Sharpness adjustment factor
        
    Returns:
        PIL.Image: Transformed image
    """
    return ImageEnhance.Sharpness(img).enhance(factor)


def _randomly_negate(v):
    """
    Randomly negate a value with 50% probability.
    
    Args:
        v (float): Value to potentially negate
        
    Returns:
        float: Possibly negated value
    """
    return -v if random.random() > 0.5 else v


# Level-to-argument conversion functions
def _rotate_level_to_arg(level, _hparams):
    # range [-30, 30]
    level = (level / _MAX_LEVEL) * 30.0
    level = _randomly_negate(level)
    return (level,)


def _enhance_level_to_arg(level, _hparams):
    # range [0.1, 1.9]
    return ((level / _MAX_LEVEL) * 1.8 + 0.1,)


def _enhance_increasing_level_to_arg(level, _hparams):
    # range [0.1, 1.9]
    level = (level / _MAX_LEVEL) * 0.9
    level = 1.0 + _randomly_negate(level)
    return (level,)


def _shear_level_to_arg(level, _hparams):
    # range [-0.3, 0.3]
    level = (level / _MAX_LEVEL) * 0.3
    level = _randomly_negate(level)
    return (level,)


def _translate_abs_level_to_arg(level, hparams):
    translate_const = hparams["translate_const"]
    level = (level / _MAX_LEVEL) * float(translate_const)
    level = _randomly_negate(level)
    return (level,)


def _translate_rel_level_to_arg(level, hparams):
    # default range [-0.45, 0.45]
    translate_pct = hparams.get("translate_pct", 0.45)
    level = (level / _MAX_LEVEL) * translate_pct
    level = _randomly_negate(level)
    return (level,)


def _posterize_level_to_arg(level, _hparams):
    # range [0, 4]
    return (int((level / _MAX_LEVEL) * 4),)


def _posterize_increasing_level_to_arg(level, hparams):
    # range [4, 0]
    return (4 - _posterize_level_to_arg(level, hparams)[0],)


def _posterize_original_level_to_arg(level, _hparams):
    # range [4, 8]
    return (int((level / _MAX_LEVEL) * 4) + 4,)


def _solarize_level_to_arg(level, _hparams):
    # range [0, 256]
    return (int((level / _MAX_LEVEL) * 256),)


def _solarize_increasing_level_to_arg(level, _hparams):
    # range [0, 256]
    return (256 - _solarize_level_to_arg(level, _hparams)[0],)


def _solarize_add_level_to_arg(level, _hparams):
    # range [0, 110]
    return (int((level / _MAX_LEVEL) * 110),)


# Mappings from operation names to level-to-argument converters
LEVEL_TO_ARG = {
    "AutoContrast": None,
    "Equalize": None,
    "Invert": None,
    "Rotate": _rotate_level_to_arg,
    "Posterize": _posterize_level_to_arg,
    "PosterizeIncreasing": _posterize_increasing_level_to_arg,
    "PosterizeOriginal": _posterize_original_level_to_arg,
    "Solarize": _solarize_level_to_arg,
    "SolarizeIncreasing": _solarize_increasing_level_to_arg,
    "SolarizeAdd": _solarize_add_level_to_arg,
    "Color": _enhance_level_to_arg,
    "ColorIncreasing": _enhance_increasing_level_to_arg,
    "Contrast": _enhance_level_to_arg,
    "ContrastIncreasing": _enhance_increasing_level_to_arg,
    "Brightness": _enhance_level_to_arg,
    "BrightnessIncreasing": _enhance_increasing_level_to_arg,
    "Sharpness": _enhance_level_to_arg,
    "SharpnessIncreasing": _enhance_increasing_level_to_arg,
    "ShearX": _shear_level_to_arg,
    "ShearY": _shear_level_to_arg,
    "TranslateX": _translate_abs_level_to_arg,
    "TranslateY": _translate_abs_level_to_arg,
    "TranslateXRel": _translate_rel_level_to_arg,
    "TranslateYRel": _translate_rel_level_to_arg,
}

# Mappings from operation names to functions
NAME_TO_OP = {
    "AutoContrast": auto_contrast,
    "Equalize": equalize,
    "Invert": invert,
    "Rotate": rotate,
    "Posterize": posterize,
    "PosterizeIncreasing": posterize,
    "PosterizeOriginal": posterize,
    "Solarize": solarize,
    "SolarizeIncreasing": solarize,
    "SolarizeAdd": solarize_add,
    "Color": color,
    "ColorIncreasing": color,
    "Contrast": contrast,
    "ContrastIncreasing": contrast,
    "Brightness": brightness,
    "BrightnessIncreasing": brightness,
    "Sharpness": sharpness,
    "SharpnessIncreasing": sharpness,
    "ShearX": shear_x,
    "ShearY": shear_y,
    "TranslateX": translate_x_abs,
    "TranslateY": translate_y_abs,
    "TranslateXRel": translate_x_rel,
    "TranslateYRel": translate_y_rel,
}


class AugmentOp:
    """
    Augmentation operation for video or image data.
    
    This class applies a specific augmentation operation with configurable
    probability and magnitude.
    
    Args:
        name (str): Name of the augmentation operation
        prob (float): Probability of applying the augmentation
        magnitude (int): Magnitude of the augmentation effect (0-10)
        hparams (dict): Hyperparameters for augmentation
    """

    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = {
            "fillcolor": hparams["img_mean"] if "img_mean" in hparams else _FILL,
            "resample": hparams["interpolation"] if "interpolation" in hparams else _RANDOM_INTERPOLATION,
        }
        
        # Randomness in magnitude
        self.magnitude_std = self.hparams.get("magnitude_std", 0)

    def __call__(self, img_list):
        """
        Apply the augmentation operation.
        
        Args:
            img_list (PIL.Image or list): Single image or list of images to augment
            
        Returns:
            Same type as input: Augmented image(s)
        """
        # Skip augmentation with probability (1-prob)
        if self.prob < 1.0 and random.random() > self.prob:
            return img_list
            
        # Calculate magnitude, possibly with randomness
        magnitude = self.magnitude
        if self.magnitude_std and self.magnitude_std > 0:
            magnitude = random.gauss(magnitude, self.magnitude_std)
        magnitude = min(_MAX_LEVEL, max(0, magnitude))  # clip to valid range
        
        # Get level arguments based on magnitude
        level_args = (
            self.level_fn(magnitude, self.hparams)
            if self.level_fn is not None
            else ()
        )

        # Apply augmentation to single image or list of images
        if isinstance(img_list, list):
            return [
                self.aug_fn(img, *level_args, **self.kwargs) for img in img_list
            ]
        else:
            return self.aug_fn(img_list, *level_args, **self.kwargs)


# Transformation sets
_RAND_TRANSFORMS = [
    "AutoContrast",
    "Equalize",
    "Invert",
    "Rotate",
    "Posterize",
    "Solarize",
    "SolarizeAdd",
    "Color",
    "Contrast",
    "Brightness",
    "Sharpness",
    "ShearX",
    "ShearY",
    "TranslateXRel",
    "TranslateYRel",
]

_RAND_INCREASING_TRANSFORMS = [
    "AutoContrast",
    "Equalize",
    "Invert",
    "Rotate",
    "PosterizeIncreasing",
    "SolarizeIncreasing",
    "SolarizeAdd",
    "ColorIncreasing",
    "ContrastIncreasing",
    "BrightnessIncreasing",
    "SharpnessIncreasing",
    "ShearX",
    "ShearY",
    "TranslateXRel",
    "TranslateYRel",
]

# Experimental weights for operations
_RAND_CHOICE_WEIGHTS_0 = {
    "Rotate": 0.3,
    "ShearX": 0.2,
    "ShearY": 0.2,
    "TranslateXRel": 0.1,
    "TranslateYRel": 0.1,
    "Color": 0.025,
    "Sharpness": 0.025,
    "AutoContrast": 0.025,
    "Solarize": 0.005,
    "SolarizeAdd": 0.005,
    "Contrast": 0.005,
    "Brightness": 0.005,
    "Equalize": 0.005,
    "Posterize": 0,
    "Invert": 0,
}


def _select_rand_weights(weight_idx=0, transforms=None):
    """
    Select probability weights for random transforms.
    
    Args:
        weight_idx (int): Index of weight set to use
        transforms (list): List of transform names
        
    Returns:
        np.ndarray: Normalized probability weights
    """
    transforms = transforms or _RAND_TRANSFORMS
    assert weight_idx == 0  # currently only one set of weights is supported
    rand_weights = _RAND_CHOICE_WEIGHTS_0
    probs = [rand_weights[k] for k in transforms]
    probs = np.array(probs)
    probs /= np.sum(probs)
    return probs


def rand_augment_ops(magnitude=10, hparams=None, transforms=None):
    """
    Create a list of augmentation operations.
    
    Args:
        magnitude (int): Magnitude of augmentation effects
        hparams (dict): Hyperparameters for augmentations
        transforms (list): List of transform names to use
        
    Returns:
        list: List of AugmentOp instances
    """
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS
    return [
        AugmentOp(name, prob=0.5, magnitude=magnitude, hparams=hparams)
        for name in transforms
    ]


class RandAugment:
    """
    RandAugment implementation.
    
    Randomly selects and applies augmentation operations to images.
    
    Args:
        ops (list): List of possible augmentation operations
        num_layers (int): Number of augmentations to apply per image
        choice_weights (np.ndarray): Probability weights for choosing operations
    """

    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def __call__(self, img):
        """
        Apply random augmentations to an image.
        
        Args:
            img (PIL.Image): Image to augment
            
        Returns:
            PIL.Image: Augmented image
        """
        # Select random operations
        ops = np.random.choice(
            self.ops,
            self.num_layers,
            replace=self.choice_weights is None,
            p=self.choice_weights,
        )
        
        # Apply operations sequentially
        for op in ops:
            img = op(img)
        return img


def rand_augment_transform(config_str, hparams):
    """
    Create a RandAugment transform from a configuration string.
    
    Args:
        config_str (str): Configuration string for RandAugment
        hparams (dict): Hyperparameters for augmentations
        
    Returns:
        RandAugment: Configured RandAugment transform
        
    Example:
        rand-m9-n3-mstd0.5 -> RandAugment with magnitude 9, num_layers 3, magnitude_std 0.5
    """
    magnitude = _MAX_LEVEL  # default to _MAX_LEVEL for magnitude (currently 10)
    num_layers = 2  # default to 2 ops per image
    weight_idx = None  # default to no probability weights for op choice
    transforms = _RAND_TRANSFORMS
    
    # Parse config string
    config = config_str.split("-")
    assert config[0] == "rand"
    config = config[1:]
    
    for c in config:
        cs = re.split(r"(\d.*)", c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == "mstd":
            # noise param injected via hparams for now
            hparams.setdefault("magnitude_std", float(val))
        elif key == "inc":
            if bool(val):
                transforms = _RAND_INCREASING_TRANSFORMS
        elif key == "m":
            magnitude = int(val)
        elif key == "n":
            num_layers = int(val)
        elif key == "w":
            weight_idx = int(val)
        else:
            assert NotImplementedError
            
    # Create operations and choice weights
    ra_ops = rand_augment_ops(
        magnitude=magnitude, hparams=hparams, transforms=transforms
    )
    choice_weights = (
        None if weight_idx is None else _select_rand_weights(weight_idx)
    )
    
    return RandAugment(ra_ops, num_layers, choice_weights=choice_weights)