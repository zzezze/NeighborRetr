#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DiDeMo dataset implementation for NeighborRetr.

This module provides the dataset class for the DiDeMo video-text retrieval dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import json
import numpy as np
from torch.utils.data import Dataset
from .rawvideo_util import RawVideoExtractor


class DiDeMoDataset(Dataset):
    """
    DiDeMo dataset for video-text retrieval.
    
    This class handles loading and processing the Distinct Describable Moments (DiDeMo) dataset
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
        Initialize the DiDeMo dataset.
        
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

        video_json_path_dict = {
            "train": os.path.join(self.data_path, "train_data.json"),
            "val": os.path.join(self.data_path, "val_data.json"),
            "test": os.path.join(self.data_path, "test_data.json")
        }
        
        # Load video IDs
        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [item.strip() for item in fp.readlines()]
   
        # Load and process annotations
        caption_dict = {}
        with open(video_json_path_dict[self.subset], 'r') as f:
            json_data = json.load(f)
            
        for item in json_data:
            description = item["description"]
            times = item["times"]
            video = item["video"]
            
            if video not in video_ids:
                continue
                
            # Each video is split into 5-second temporal chunks
            # Average the points from each annotator
            start_ = np.mean([t_[0] for t_ in times]) * 5
            end_ = (np.mean([t_[1] for t_ in times]) + 1) * 5
            
            if video in caption_dict:
                caption_dict[video]["start"].append(start_)
                caption_dict[video]["end"].append(end_)
                caption_dict[video]["text"].append(description)
            else:
                caption_dict[video] = {}
                caption_dict[video]["start"] = [start_]
                caption_dict[video]["end"] = [end_]
                caption_dict[video]["text"] = [description]

        # Simplify captions for efficiency
        for k_ in caption_dict.keys():
            caption_dict[k_]["start"] = [0]
            # Trick to save time on obtaining each video length
            # [https://github.com/LisaAnne/LocalizingMoments/blob/master/README.md]:
            # Some videos are longer than 30 seconds. These videos were truncated to 30 seconds during annotation.
            caption_dict[k_]["end"] = [31]
            caption_dict[k_]["text"] = [" ".join(caption_dict[k_]["text"])]

        # Create video dictionary mapping IDs to file paths
        video_dict = {}
        for root, _, video_files in os.walk(self.features_path):
            for video_file in video_files:
                video_id_ = os.path.splitext(video_file)[0]
                if video_id_ not in video_ids:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_

        # Store dictionaries
        self.caption_dict = caption_dict
        self.video_dict = video_dict
        
        # Filter video IDs to only include those with captions and features
        video_ids = list(set(video_ids) & set(self.caption_dict.keys()) & set(self.video_dict.keys()))

        # Create video-caption pairs
        self.iter2video_pairs_dict = {}
        for video_id in self.caption_dict.keys():
            if video_id not in video_ids:
                continue
                
            caption = self.caption_dict[video_id]
            n_caption = len(caption['start'])
            
            for sub_id in range(n_caption):
                self.iter2video_pairs_dict[len(self.iter2video_pairs_dict)] = (
                    video_id, sub_id
                )

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
        return len(self.iter2video_pairs_dict)

    def _get_text(self, video_id, sub_id):
        """
        Process text captions for a video.
        
        Args:
            video_id (str): Video ID
            sub_id (int): Caption index
            
        Returns:
            tuple: (pairs_text, pairs_mask, pairs_segment, starts, ends)
        """
        caption = self.caption_dict[video_id]
        k = 1  # Number of captions per video
        r_ind = [sub_id]

        # Initialize arrays
        starts = np.zeros(k, dtype=np.long)
        ends = np.zeros(k, dtype=np.long)
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i in range(k):
            ind = r_ind[i]
            start_, end_ = caption['start'][ind], caption['end'][ind]
            words = self.tokenizer.tokenize(caption['text'][ind])
            starts[i], ends[i] = start_, end_

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

        return pairs_text, pairs_mask, pairs_segment, starts, ends

    def _get_rawvideo(self, idx, s, e):
        """
        Get video frames for a video.
        
        Args:
            idx (str): Video ID
            s (np.array): Start times
            e (np.array): End times
            
        Returns:
            tuple: (video, video_mask)
        """
        # Initialize arrays
        video_mask = np.zeros((len(s), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(s)

        # Create empty video tensor
        video = np.zeros(
            (len(s), self.max_frames, 1, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size), 
            dtype=np.float
        )
        
        video_path = self.video_dict[idx]

        try:
            for i in range(len(s)):
                # Process time boundaries
                start_time = int(s[i])
                end_time = int(e[i])
                start_time = max(start_time, 0)
                end_time = max(end_time, 0)
                
                if start_time > end_time:
                    start_time, end_time = end_time, start_time
                elif start_time == end_time:
                    end_time = end_time + 1

                # Create cache ID for potential future optimization
                cache_id = f"{video_path}_{start_time}_{end_time}"
                
                # Extract video data
                raw_video_data = self.rawVideoExtractor.get_video_data(
                    video_path, start_time, end_time
                )
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
                    print(f"Video path: {video_path} error. Video id: {idx}, "
                          f"start: {start_time}, end: {end_time}")
                          
        except Exception as excep:
            print(f"Video path: {video_path} error. Video id: {idx}, "
                  f"start: {s}, end: {e}, Error: {excep}")
            # Log error but don't propagate exception to allow processing to continue
            pass

        # Create mask for valid frames
        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, feature_idx):
        """
        Get a video-text pair.
        
        Args:
            feature_idx (int): Index of the item
            
        Returns:
            tuple: (pairs_text, pairs_mask, video, video_mask, feature_idx, hash(video_id))
        """
        video_id, sub_id = self.iter2video_pairs_dict[feature_idx]

        # Get text and video features
        pairs_text, pairs_mask, pairs_segment, starts, ends = self._get_text(
            video_id, sub_id
        )
        video, video_mask = self._get_rawvideo(
            video_id, starts, ends
        )
        
        return pairs_text, pairs_mask, video, video_mask, feature_idx, hash(video_id)