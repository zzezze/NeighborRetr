#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSVD dataset implementation for NeighborRetr.

This module provides the dataset class for the Microsoft Video Description (MSVD) 
video-text retrieval dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from .rawvideo_util import RawVideoExtractor


class MsvdDataset(Dataset):
    """
    MSVD dataset for video-text retrieval.
    
    This class handles loading and processing the Microsoft Video Description (MSVD) dataset
    with efficient caching and data loading.
    """

    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=2,
    ):
        """
        Initialize the MSVD dataset.
        
        Args:
            subset (str): Dataset split ('train', 'val', 'test')
            data_path (str): Path to annotations
            features_path (str): Path to video features
            tokenizer: Tokenizer for text processing
            max_words (int): Maximum number of words in text
            feature_framerate (float): Frame rate for video sampling
            max_frames (int): Maximum number of video frames
            image_resolution (int): Resolution for image/video frames
            frame_order (int): Order of frames (0: normal, 1: reverse, 2: random)
            slice_framepos (int): Frame sampling strategy (0: head, 1: tail, 2: uniform)
        """
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        
        # Frame ordering strategy
        # 0: ordinary order; 1: reverse order; 2: random order
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        
        # Frame sampling strategy
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        # Dataset subset
        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        
        # Define paths based on subset
        video_id_path_dict = {
            "train": os.path.join(self.data_path, "train_list.txt"),
            "val": os.path.join(self.data_path, "val_list.txt"),
            "test": os.path.join(self.data_path, "test_list.txt")
        }
        
        caption_file = os.path.join(self.data_path, "raw-captions.pkl")

        # Load video IDs
        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [item.strip() for item in fp.readlines()]

        # Load captions
        with open(caption_file, 'rb') as f:
            captions = pickle.load(f)

        # Create video dictionary mapping IDs to file paths
        video_dict = {}
        for root, _, video_files in os.walk(self.features_path):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])
                if video_id_ not in video_ids:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_
                
        self.video_dict = video_dict

        # Create sentence dictionary for video-caption pairs
        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        
        for video_id in video_ids:
            assert video_id in captions
            for cap in captions[video_id]:
                cap_txt = " ".join(cap)
                self.sentences_dict[len(self.sentences_dict)] = (video_id, cap_txt)
            self.cut_off_points.append(len(self.sentences_dict))

        # Variables used for multi-sentence retrieval evaluation
        # self.cut_off_points: used to tag the label when calculating metrics
        # self.sentence_num: used to cut the sentence representation
        # self.video_num: used to cut the video representation
        self.multi_sentence_per_video = True  # Important tag for evaluation
        
        if self.subset in ["train", "val", "test"]:
            self.sentence_num = len(self.sentences_dict)
            self.video_num = len(video_ids)
            assert len(self.cut_off_points) == self.video_num
            print(f"For {self.subset}, sentence number: {self.sentence_num}")
            print(f"For {self.subset}, video number: {self.video_num}")

        print(f"Video number: {len(self.video_dict)}")
        print(f"Total Pairs: {len(self.sentences_dict)}")

        self.sample_len = len(self.sentences_dict)
        
        # Initialize video extractor and special tokens
        self.rawVideoExtractor = RawVideoExtractor(
            framerate=feature_framerate, 
            size=image_resolution
        )
        
        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>", 
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]", 
            "UNK_TOKEN": "[UNK]", 
            "PAD_TOKEN": "[PAD]"
        }

    def __len__(self):
        """Get the length of the dataset"""
        return self.sample_len

    def _get_text(self, video_id, caption):
        """
        Process text captions for a video.
        
        Args:
            video_id (str): Video ID
            caption (str): Caption text
            
        Returns:
            tuple: (pairs_text, pairs_mask, pairs_segment, choice_video_ids)
        """
        k = 1  # Number of captions per video
        choice_video_ids = [video_id]
        
        # Initialize arrays
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, vid in enumerate(choice_video_ids):
            # Tokenize caption
            words = self.tokenizer.tokenize(caption)

            # Add special tokens and truncate if needed
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            # Convert tokens to IDs and create mask
            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            
            # Pad to max length
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                
            # Verify lengths
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            # Store processed text data
            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        """
        Get video frames for videos.
        
        Args:
            choice_video_ids (list): List of video IDs
            
        Returns:
            tuple: (video, video_mask)
        """
        # Initialize arrays
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Create empty video tensor
        video = np.zeros(
            (len(choice_video_ids), self.max_frames, 1, 3, 
             self.rawVideoExtractor.size, self.rawVideoExtractor.size), 
            dtype=np.float
        )

        for i, video_id in enumerate(choice_video_ids):
            video_path = self.video_dict[video_id]

            # Extract full video (no time boundaries for MSVD)
            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                # Process video frames
                raw_video_data_clip = raw_video_data
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                
                # Apply frame sampling strategy
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(
                            0, raw_video_slice.shape[0] - 1, 
                            num=self.max_frames, 
                            dtype=int
                        )
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                # Apply frame ordering strategy
                video_slice = self.rawVideoExtractor.process_frame_order(
                    video_slice, 
                    frame_order=self.frame_order
                )

                # Store video data
                slice_len = video_slice.shape[0]
                max_video_length[i] = max(max_video_length[i], slice_len)
                
                if slice_len > 0:
                    video[i][:slice_len, ...] = video_slice
            else:
                print(f"Video path: {video_path} error. Video id: {video_id}")

        # Create mask for valid frames
        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        """
        Get a video-text pair.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            tuple: (pairs_text, pairs_mask, video, video_mask, idx, hash(video_id))
        """
        video_id, caption = self.sentences_dict[idx]

        # Get text and video features
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(
            video_id, caption
        )
        video, video_mask = self._get_rawvideo(choice_video_ids)
        
        return pairs_text, pairs_mask, video, video_mask, idx, hash(video_id)