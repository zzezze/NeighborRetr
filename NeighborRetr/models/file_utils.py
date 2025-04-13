#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities for working with the local dataset cache.

This module provides functions for file caching, downloading, 
and handling URLs for pre-trained models and datasets.

Adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""

import os
import logging
import shutil
import tempfile
import json
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional, Tuple, Union, IO, Callable, Set
from hashlib import sha256
from functools import wraps

from tqdm import tqdm

import boto3
from botocore.exceptions import ClientError
import requests

logger = logging.getLogger(__name__)

# Default cache directory for pretrained models
PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv(
    'PYTORCH_PRETRAINED_BERT_CACHE',
    Path.home() / '.pytorch_pretrained_bert'))


def url_to_filename(url: str, etag: str = None) -> str:
    """
    Convert URL into a hashed filename in a repeatable way.
    
    Args:
        url: URL to convert to filename
        etag: Optional ETag for the URL for versioning
        
    Returns:
        Hashed filename string
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()

    return filename


def filename_to_url(filename: str, cache_dir: Union[str, Path] = None) -> Tuple[str, str]:
    """
    Return the URL and ETag stored for a cached filename.
    
    Args:
        filename: Filename to lookup
        cache_dir: Directory to search for the file
        
    Returns:
        Tuple of (url, etag)
        
    Raises:
        FileNotFoundError: If file or metadata doesn't exist
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"file {cache_path} not found")

    meta_path = cache_path + '.json'
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"file {meta_path} not found")

    with open(meta_path) as meta_file:
        metadata = json.load(meta_file)
    url = metadata['url']
    etag = metadata['etag']

    return url, etag


def cached_path(url_or_filename: Union[str, Path], cache_dir: Union[str, Path] = None) -> str:
    """
    Get path to cached or local file.
    
    Given something that might be a URL or local path, determine which it is.
    If it's a URL, download and cache the file, then return the path.
    If it's already a local path, verify the file exists and return the path.
    
    Args:
        url_or_filename: URL or path to a file
        cache_dir: Directory to store cached files
        
    Returns:
        Path to the cached or local file
        
    Raises:
        FileNotFoundError: If the file doesn't exist locally
        ValueError: If the input isn't a URL or local path
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ('http', 'https', 's3'):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        # File exists locally
        return url_or_filename
    elif parsed.scheme == '':
        # File doesn't exist
        raise FileNotFoundError(f"file {url_or_filename} not found")
    else:
        # Unknown scheme
        raise ValueError(f"unable to parse {url_or_filename} as a URL or local path")


def split_s3_path(url: str) -> Tuple[str, str]:
    """
    Split an S3 path into bucket name and path.
    
    Args:
        url: S3 URL
        
    Returns:
        Tuple of (bucket_name, s3_path)
        
    Raises:
        ValueError: If the URL is not a valid S3 path
    """
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError(f"bad s3 path {url}")
    bucket_name = parsed.netloc
    s3_path = parsed.path
    # Remove leading slash
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


def s3_request(func: Callable) -> Callable:
    """
    Wrapper for S3 requests to provide more helpful error messages.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(url: str, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise FileNotFoundError(f"file {url} not found")
            else:
                raise
    return wrapper


@s3_request
def s3_etag(url: str) -> Optional[str]:
    """
    Get ETag for an S3 object.
    
    Args:
        url: S3 URL
        
    Returns:
        ETag if available, else None
    """
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@s3_request
def s3_get(url: str, temp_file: IO) -> None:
    """
    Download a file from S3.
    
    Args:
        url: S3 URL
        temp_file: File-like object to write to
    """
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def http_get(url: str, temp_file: IO) -> None:
    """
    Download a file from HTTP(S).
    
    Args:
        url: HTTP(S) URL
        temp_file: File-like object to write to
    """
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # Filter out keep-alive chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(url: str, cache_dir: Union[str, Path] = None) -> str:
    """
    Download and cache a file from a URL.
    
    Args:
        url: URL to download
        cache_dir: Directory to store cached files
        
    Returns:
        Path to the cached file
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Get ETag for versioning
    if url.startswith("s3://"):
        etag = s3_etag(url)
    else:
        response = requests.head(url, allow_redirects=True)
        if response.status_code != 200:
            raise IOError(f"HEAD request failed for url {url} with status code {response.status_code}")
        etag = response.headers.get("ETag")

    # Create filename based on URL and ETag
    filename = url_to_filename(url, etag)
    cache_path = os.path.join(cache_dir, filename)

    if not os.path.exists(cache_path):
        # Download to temporary file first to avoid corruption
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info(f"{url} not found in cache, downloading to {temp_file.name}")

            # Download file
            if url.startswith("s3://"):
                s3_get(url, temp_file)
            else:
                http_get(url, temp_file)

            # Flush to ensure all data is written
            temp_file.flush()
            # Move to beginning of file for copying
            temp_file.seek(0)

            logger.info(f"copying {temp_file.name} to cache at {cache_path}")
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            # Save metadata
            logger.info(f"creating metadata file for {cache_path}")
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'
            with open(meta_path, 'w') as meta_file:
                json.dump(meta, meta_file)

            logger.info(f"removing temp file {temp_file.name}")

    return cache_path


def read_set_from_file(filename: str) -> Set[str]:
    """
    Read a set of lines from a file.
    
    Args:
        filename: Path to the file
        
    Returns:
        Set of strings, one per line in the file
    """
    collection = set()
    with open(filename, 'r', encoding='utf-8') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection


def get_file_extension(path: str, dot: bool = True, lower: bool = True) -> str:
    """
    Get the file extension from a path.
    
    Args:
        path: Path to a file
        dot: Whether to include the dot in the extension
        lower: Whether to lowercase the extension
        
    Returns:
        File extension string
    """
    ext = os.path.splitext(path)[1]
    ext = ext if dot else ext[1:]
    return ext.lower() if lower else ext