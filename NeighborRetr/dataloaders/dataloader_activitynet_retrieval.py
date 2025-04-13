#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ActivityNet dataset implementation for NeighborRetr.

This module provides the dataset class for the ActivityNet video-text retrieval dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import json
import math
import numpy as np
from torch.utils.data import Dataset
from .rawvideo_util import RawVideoExtractor


class ActivityNetDataset(Dataset):
    """
    ActivityNet dataset for video-text retrieval.
    
    This class handles loading and processing the ActivityNet dataset
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
        Initialize the ActivityNet dataset.
        
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
        assert self.subset in ["train", "val"]

        # Define paths for different subsets
        video_id_path_dict = {
            "train": os.path.join(self.data_path, "train_ids.json"),
            "val": os.path.join(self.data_path, "val_ids.json")
        }

        video_json_path_dict = {
            "train": os.path.join(self.data_path, "train.json"),
            "val": os.path.join(self.data_path, "val_1.json")
        }

        # Load video IDs and captions
        pseudo_video_id_list, video_id_list = self._get_video_id_single(
            video_id_path_dict[self.subset]
        )
        pseudo_caption_dict = self._get_captions_single(
            video_json_path_dict[self.subset]
        )

        print(f"Video ID list: {len(video_id_list)}")
        print(f"Pseudo caption dict: {len(pseudo_caption_dict.keys())}")

        # Create video dictionary mapping IDs to file paths
        video_dict = {}
        for root, _, video_files in os.walk(self.features_path):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])[2:]
                if video_id_ not in video_id_list:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_
                
        self.video_dict = video_dict
        print(f"Video dict: {len(video_dict)}")

        # Store lists for retrieval
        self.pseudo_video_id_list = pseudo_video_id_list
        self.video_id_list = video_id_list
        self.pseudo_caption_dict = pseudo_caption_dict

        # Create mapping and pairing dictionaries
        self.video_id2idx_dict = {
            pseudo_video_id: idx for idx, pseudo_video_id in enumerate(self.pseudo_video_id_list)
        }
        
        # Create video-caption pairs
        self.iter2video_pairs_dict = {}
        for pseudo_video_id, video_id in zip(self.pseudo_video_id_list, self.video_id_list):
            if pseudo_video_id not in self.pseudo_caption_dict or video_id not in self.video_dict:
                continue
                
            caption = self.pseudo_caption_dict[pseudo_video_id]
            n_caption = len(caption['start'])
            
            for sub_id in range(n_caption):
                self.iter2video_pairs_dict[len(self.iter2video_pairs_dict)] = (
                    pseudo_video_id, sub_id
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

    def _get_video_id_from_pseduo(self, pseudo_video_id):
        """Extract the actual video ID from a pseudo video ID"""
        video_id = pseudo_video_id[2:]
        return video_id

    def _get_video_id_single(self, path):
        """
        Load video IDs from a JSON file.
        
        Args:
            path (str): Path to the JSON file
            
        Returns:
            tuple: (pseudo_video_id_list, video_id_list)
        """
        pseudo_video_id_list = []
        video_id_list = []
        
        print(f'Loading json: {path}')
        with open(path, 'r') as f:
            json_data = json.load(f)

        for pseudo_video_id in json_data:
            if pseudo_video_id in pseudo_video_id_list:
                print("Duplicate ID found.")
            else:
                video_id = self._get_video_id_from_pseduo(pseudo_video_id)
                pseudo_video_id_list.append(pseudo_video_id)
                video_id_list.append(video_id)
                
        return pseudo_video_id_list, video_id_list

    def _get_captions_single(self, path):
        """
        Load captions from a JSON file.
        
        Args:
            path (str): Path to the JSON file
            
        Returns:
            dict: Dictionary of captions
        """
        pseudo_caption_dict = {}
        
        with open(path, 'r') as f:
            json_data = json.load(f)

        for pseudo_video_id, v_ in json_data.items():
            pseudo_caption_dict[pseudo_video_id] = {}
            duration = v_["duration"]
            
            # Set start time, end time, and text
            pseudo_caption_dict[pseudo_video_id]["start"] = np.array([0], dtype=object)
            pseudo_caption_dict[pseudo_video_id]["end"] = np.array(
                [int(math.ceil(float(duration)))], 
                dtype=object
            )
            pseudo_caption_dict[pseudo_video_id]["text"] = np.array(
                [" ".join(v_["sentences"])], 
                dtype=object
            )
            
        return pseudo_caption_dict

    def _get_text(self, pseudo_video_id, sub_id):
        """
        Process text captions for a video.
        
        Args:
            pseudo_video_id (str): Pseudo video ID
            sub_id (int): Caption index
            
        Returns:
            tuple: (pairs_text, pairs_mask, pairs_segment, starts, ends)
        """
        caption = self.pseudo_caption_dict[pseudo_video_id]
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
            raise excep

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
            tuple: (pairs_text, pairs_mask, video, video_mask, feature_idx, hash(pseudo_video_id))
        """
        pseudo_video_id, sub_id = self.iter2video_pairs_dict[feature_idx]
        idx = self.video_id2idx_dict[pseudo_video_id]

        # Get text and video features
        pairs_text, pairs_mask, pairs_segment, starts, ends = self._get_text(
            pseudo_video_id, sub_id
        )
        video, video_mask = self._get_rawvideo(
            self.video_id_list[idx], starts, ends
        )
        
        return pairs_text, pairs_mask, video, video_mask, feature_idx, hash(pseudo_video_id)


def load_stopwords(path='data/english.txt'):
    """
    Load stopwords from a file.
    
    Args:
        path (str): Path to stopwords file
        
    Returns:
        list: List of stopwords
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def remove_stopwords(documents, stopwords):
    """
    Remove stopwords from a document.
    
    Args:
        documents (str): Document text
        stopwords (list): List of stopwords
        
    Returns:
        str: Document with stopwords removed
    """
    cleaned_documents = []
    for token in documents.split():
        if token not in stopwords:
            cleaned_documents.append(token)
    return " ".join('%s' % a for a in cleaned_documents)