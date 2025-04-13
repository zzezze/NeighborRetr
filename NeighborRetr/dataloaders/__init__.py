#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataloaders module for NeighborRetr.

This package contains dataloaders for various video-text datasets
used with the NeighborRetr model for cross-modal retrieval.
"""

from NeighborRetr.dataloaders.data_dataloaders import (
    DATALOADER_DICT,
    dataloader_msrvtt_train,
    dataloader_msrvtt_test,
    dataloader_activity_train,
    dataloader_activity_test,
    dataloader_didemo_train,
    dataloader_didemo_test,
    dataloader_msvd_train,
    dataloader_msvd_test
)

__all__ = [
    'DATALOADER_DICT',
    'dataloader_msrvtt_train',
    'dataloader_msrvtt_test',
    'dataloader_activity_train',
    'dataloader_activity_test',
    'dataloader_didemo_train',
    'dataloader_didemo_test',
    'dataloader_msvd_train',
    'dataloader_msvd_test'
]