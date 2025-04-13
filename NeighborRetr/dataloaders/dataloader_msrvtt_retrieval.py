#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSR-VTT dataset implementation for NeighborRetr.

This module provides the dataset class for the MSR-VTT video-text retrieval dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import pandas as pd
from os.path import join, exists
from collections import OrderedDict
from .dataloader_retrieval import RetrievalDataset


class MSRVTTDataset(RetrievalDataset):
    """
    MSR-VTT dataset for video-text retrieval.
    
    This class handles loading and processing the MSR-VTT dataset
    with efficient caching and data loading.
    """

    def __init__(
            self,
            subset,
            anno_path,
            video_path,
            tokenizer,
            max_words=32,
            max_frames=12,
            video_framerate=1,
            image_resolution=224,
            mode='all',
            config=None
    ):
        """
        Initialize the MSR-VTT dataset.
        
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
        super(MSRVTTDataset, self).__init__(
            subset=subset,
            anno_path=anno_path,
            video_path=video_path,
            tokenizer=tokenizer,
            max_words=max_words,
            max_frames=max_frames,
            video_framerate=video_framerate,
            image_resolution=image_resolution,
            mode=mode,
            config=config
        )

    def _get_anns(self, subset='train'):
        """
        Get annotations for the MSR-VTT dataset.
        
        Args:
            subset (str): Dataset split ('train', 'val', 'test')
            
        Returns:
            tuple: (video_dict, sentences_dict)
                video_dict (dict): video_id -> video_path
                sentences_dict (list): [(video_id, caption)]
                    where caption is (text, start_time, end_time)
                    
        Raises:
            FileNotFoundError: If annotation files don't exist
        """
        # Define paths to CSV files based on subset
        csv_path_mapping = {
            'train': join(self.anno_path, 'MSRVTT_train.9k.csv'),
            'val': join(self.anno_path, 'MSRVTT_JSFUSION_test.csv'),
            'test': join(self.anno_path, 'MSRVTT_JSFUSION_test.csv')
        }
        
        csv_path = csv_path_mapping[subset]
        
        # Load CSV file
        if exists(csv_path):
            csv = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        # Get video IDs
        video_id_list = list(csv['video_id'].values)
        
        # Initialize dictionaries for video paths and captions
        video_dict = OrderedDict()
        sentences_dict = OrderedDict()
        
        # Process annotations based on subset
        if subset == 'train':
            # Load JSON data for training
            anno_path = join(self.anno_path, 'MSRVTT_data.json')
            data = json.load(open(anno_path, 'r'))
            
            # Process each caption
            for item in data['sentences']:
                if item['video_id'] in video_id_list:
                    # Add caption to sentences dictionary
                    # Format: (video_id, (caption_text, start_time, end_time))
                    sentences_dict[len(sentences_dict)] = (
                        item['video_id'], 
                        (item['caption'], None, None)
                    )
                    
                    # Add video path to video dictionary
                    video_dict[item['video_id']] = join(
                        self.video_path, 
                        f"{item['video_id']}.mp4"
                    )
        else:
            # Process test/validation data
            for _, item in csv.iterrows():
                # Add caption to sentences dictionary
                sentences_dict[len(sentences_dict)] = (
                    item['video_id'], 
                    (item['sentence'], None, None)
                )
                
                # Add video path to video dictionary
                video_dict[item['video_id']] = join(
                    self.video_path, 
                    f"{item['video_id']}.mp4"
                )
        
        # Log statistics
        unique_sentences = set([v[1][0] for v in sentences_dict.values()])
 
        
        return video_dict, sentences_dict