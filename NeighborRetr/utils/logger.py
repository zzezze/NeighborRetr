#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains utilities for setting up and configuring logging.
"""

import logging
import os
import sys


def setup_logger(name, save_dir, dist_rank, filename="log.txt"):
    """
    Sets up a logger with console and file handlers.
    
    Args:
        name (str): Name of the logger.
        save_dir (str): Directory to save log file.
        dist_rank (int): Distributed rank of the process.
        filename (str, optional): Name of the log file. Defaults to "log.txt".
        
    Returns:
        logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)
    
    # Don't log results for non-master processes
    if dist_rank > 0:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter("[%(asctime)s %(name)s %(lineno)s %(levelname)s]: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False
    
    # Create file handler if save_dir is provided
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(os.path.join(save_dir, filename))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger