#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for NeighborRetr.
Contains helper functions for parallel processing and logging.
"""

import torch
import torch.nn as nn
import threading
from torch._utils import ExceptionWrapper
import logging


def get_a_var(obj):
    """
    Recursively find and return the first PyTorch Tensor in an object.
    
    Args:
        obj: Object to search for a tensor. Can be a tensor, list, tuple, or dict.
        
    Returns:
        torch.Tensor or None: First tensor found in the object, or None if no tensor is found.
    """
    if isinstance(obj, torch.Tensor):
        return obj
    
    if isinstance(obj, (list, tuple)):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
                
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
                
    return None


def parallel_apply(fct, model, inputs, device_ids):
    """
    Apply a function in parallel across multiple devices.
    
    Args:
        fct (callable): Function to apply.
        model (nn.Module): Model to replicate across devices.
        inputs (list): List of inputs for each device.
        device_ids (list): List of device IDs.
        
    Returns:
        list: Outputs from the function applied to each input on its corresponding device.
    """
    modules = nn.parallel.replicate(model, device_ids)
    assert len(modules) == len(inputs), "Number of modules must match number of inputs"
    
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()
    
    def _worker(i, module, input):
        torch.set_grad_enabled(grad_enabled)
        device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # This also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = fct(module, *input)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where=f"in replica {i} on device {device}"
                )
    
    if len(modules) > 1:
        threads = [
            threading.Thread(target=_worker, args=(i, module, input))
            for i, (module, input) in enumerate(zip(modules, inputs))
        ]
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0])
    
    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
        
    return outputs


def get_logger(filename=None):
    """
    Configure and return a logger.
    
    Args:
        filename (str, optional): Path to log file. If provided, logs will be written to this file.
            Defaults to None.
            
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        
    return logger