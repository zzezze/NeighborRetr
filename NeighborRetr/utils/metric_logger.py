#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains utilities for logging metrics during training.
Original code from Facebook Research.
"""

from collections import defaultdict
from collections import deque

import torch


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    
    def __init__(self, window_size=20):
        """
        Initialize SmoothedValue with a specific window size.
        
        Args:
            window_size (int, optional): Size of the moving window. Defaults to 20.
        """
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0
    
    def update(self, value):
        """
        Add a new value to the series.
        
        Args:
            value (float): Value to be added to the series.
        """
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value
    
    @property
    def median(self):
        """
        Get the median value over the window.
        
        Returns:
            float: Median value in the current window.
        """
        d = torch.tensor(list(self.deque))
        return d.median().item()
    
    @property
    def avg(self):
        """
        Get the average value over the window.
        
        Returns:
            float: Average value in the current window.
        """
        d = torch.tensor(list(self.deque))
        return d.mean().item()
    
    @property
    def global_avg(self):
        """
        Get the global average over all values.
        
        Returns:
            float: Global average of all values.
        """
        return self.total / self.count if self.count > 0 else 0.0


class MetricLogger(object):
    """
    Logger for tracking multiple metrics during training.
    """
    
    def __init__(self, delimiter="\t"):
        """
        Initialize MetricLogger with a specified delimiter.
        
        Args:
            delimiter (str, optional): Delimiter for string representation. Defaults to "\t".
        """
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
    
    def update(self, **kwargs):
        """
        Update multiple metrics.
        
        Args:
            **kwargs: Arbitrary keyword arguments of metric name and value pairs.
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int)), f"Value for {k} must be float or int, got {type(v)}"
            self.meters[k].update(v)
    
    def __getattr__(self, attr):
        """
        Access a meter by attribute name.
        
        Args:
            attr (str): Name of the meter to access.
            
        Returns:
            SmoothedValue: The requested meter.
            
        Raises:
            AttributeError: If the attribute is not found.
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
    
    def __str__(self):
        """
        Get string representation of all meters.
        
        Returns:
            str: Formatted string with all metrics.
        """
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                f"{name}: {meter.median:.4f} ({meter.global_avg:.4f})"
            )
        return self.delimiter.join(loss_str)