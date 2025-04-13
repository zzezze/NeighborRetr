#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video compression utility for NeighborRetr preprocessing.
Compresses videos for faster processing using FFMPEG.
"""

import os
import argparse
import subprocess
import shutil
from multiprocessing import Pool

# Try to use psutil for CPU count if available, otherwise use multiprocessing
try:
    from psutil import cpu_count
except ImportError:
    from multiprocessing import cpu_count


def compress(paras):
    """
    Compress a video using ffmpeg.
    
    Args:
        paras (tuple): A tuple containing (input_video_path, output_video_path).
        
    Raises:
        Exception: If any error occurs during compression.
    """
    input_video_path, output_video_path = paras
    try:
        command = [
            'ffmpeg',
            '-y',  # (optional) overwrite output file if it exists
            '-i', input_video_path,
            '-filter:v',
            # Scale to 224 while maintaining aspect ratio
            'scale=\'if(gt(a,1),trunc(oh*a/2)*2,224)\':\'if(gt(a,1),224,trunc(ow*a/2)*2)\'',
            '-map', '0:v',
            '-r', '3',  # frames per second
            output_video_path,
        ]
        ffmpeg = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        _, _ = ffmpeg.communicate()
        ffmpeg.poll()
    except Exception as e:
        raise e


def prepare_input_output_pairs(input_root, output_root):
    """
    Prepare pairs of input and output video paths for processing.
    
    Args:
        input_root (str): Directory containing input videos.
        output_root (str): Directory for output compressed videos.
        
    Returns:
        tuple: Two lists containing input_video_paths and corresponding output_video_paths.
    """
    input_video_path_list = []
    output_video_path_list = []
    
    for root, _, files in os.walk(input_root):
        for file_name in files:
            input_video_path = os.path.join(root, file_name)
            output_video_path = os.path.join(output_root, file_name)
            
            # Skip if the output already exists and is not empty
            if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                continue
            
            input_video_path_list.append(input_video_path)
            output_video_path_list.append(output_video_path)
            
    return input_video_path_list, output_video_path_list


def main():
    """Main function to handle command-line arguments and process videos."""
    parser = argparse.ArgumentParser(description='Compress video for speed-up')
    parser.add_argument('--input_root', type=str, required=True, help='input root directory')
    parser.add_argument('--output_root', type=str, required=True, help='output root directory')
    args = parser.parse_args()
    
    input_root = args.input_root
    output_root = args.output_root
    
    # Safety check: input and output directories must be different
    assert input_root != output_root, "Input and output directories must be different"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)
    
    # Get list of videos to process
    input_video_path_list, output_video_path_list = prepare_input_output_pairs(input_root, output_root)
    
    print(f"Total videos to process: {len(input_video_path_list)}")
    
    # Use all available CPU cores for parallel processing
    num_workers = cpu_count()
    print(f"Beginning with {num_workers}-core logical processor.")
    
    # Process videos in parallel
    pool = Pool(num_workers)
    pool.map(
        compress,
        [(input_path, output_path) 
         for input_path, output_path in zip(input_video_path_list, output_video_path_list)]
    )
    pool.close()
    pool.join()
    
    # Check for and fix any failed compressions
    print("Compression finished, checking for failed compressions...")
    for input_video_path, output_video_path in zip(input_video_path_list, output_video_path_list):
        if os.path.exists(input_video_path):
            if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) < 1:
                shutil.copyfile(input_video_path, output_video_path)
                print(f"Copied and replaced failed file: {output_video_path}")


if __name__ == "__main__":
    main()