#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metrics functionality for NeighborRetr.

This module provides a RetrievalMetrics class to compute, print, and track the best 
retrieval metrics for text-to-video and video-to-text retrieval tasks.
"""
import numpy as np
import torch
from typing import Dict, Tuple, List, Optional, Union


class RetrievalMetrics:
    """
    A class to handle retrieval metrics computation, tracking, and reporting.
    
    This class maintains the state of the best metrics across training and evaluation
    and provides methods to update them when new evaluation results are available.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the RetrievalMetrics tracker.
        
        Args:
            logger: Logger object for printing information
        """
        # 初始化最佳得分指标
        # Initialize best metric scores with small positive values to ensure first valid results will update them
        self.best_mean_r1 = 0.00001
        self.best_t2v_r1 = 0.00001
        self.best_v2t_r1 = 0.00001
        self.best_t2v_metrics = None
        self.best_v2t_metrics = None
        self.logger = logger
    
    @staticmethod
    def compute_metrics(similarity_matrix: np.ndarray) -> Dict[str, float]:
        """
        Compute retrieval metrics from a similarity matrix.
        
        Args:
            similarity_matrix: Matrix of similarity scores between queries and targets
            
        Returns:
            dict: Dictionary containing retrieval metrics:
                - 'R1': Recall@1 percentage
                - 'R5': Recall@5 percentage
                - 'R10': Recall@10 percentage
                - 'R50': Recall@50 percentage
                - 'MR': Median rank
                - 'MedianR': Same as MR (for compatibility)
                - 'MeanR': Mean rank
                - 'cols': List of ranks for each query
        """
        # Sort similarity scores in descending order
        sx = np.sort(-similarity_matrix, axis=1)
        # Get diagonal elements (matching pairs)
        d = np.diag(-similarity_matrix)
        d = d[:, np.newaxis]
        # Calculate difference between sorted scores and diagonal
        ind = sx - d
        # Find where the difference is 0 (indicates rank position)
        ind = np.where(ind == 0)
        ind = ind[1]
        
        # Calculate metrics
        metrics = {}
        metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
        metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
        metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
        metrics['R50'] = float(np.sum(ind < 50)) * 100 / len(ind)
        metrics['MR'] = float(np.median(ind)) + 1
        metrics["MedianR"] = metrics['MR']
        metrics["MeanR"] = float(np.mean(ind)) + 1
        metrics["cols"] = [int(i) for i in list(ind)]
        
        return metrics
    
    @staticmethod
    def tensor_text_to_video_metrics(sim_tensor: Union[np.ndarray, torch.Tensor], 
                                    top_k: List[int] = [1, 5, 10, 50]) -> Dict[str, float]:
        """
        Compute text-to-video retrieval metrics from a 3D similarity tensor.
        
        This function handles the more complex case of multiple texts per video.
        
        Args:
            sim_tensor: 3D tensor of similarity scores [text_batch, video_batch, text_per_video]
            top_k: List of k values for Recall@k calculation
            
        Returns:
            dict: Dictionary containing retrieval metrics
        """
        if not torch.is_tensor(sim_tensor):
            sim_tensor = torch.tensor(sim_tensor)
            
        # Permute sim_tensor so it represents a sequence of text-video similarity matrices.
        # Then obtain the double argsort to position the rank on the diagonal
        stacked_sim_matrices = sim_tensor.permute(1, 0, 2)
        first_argsort = torch.argsort(stacked_sim_matrices, dim=-1, descending=True)
        second_argsort = torch.argsort(first_argsort, dim=-1, descending=False)
        
        # Extract ranks (diagonals)
        ranks = torch.flatten(torch.diagonal(second_argsort, dim1=1, dim2=2))
        
        # Filter out invalid ranks (inf or nan values)
        permuted_original_data = torch.flatten(torch.diagonal(sim_tensor, dim1=0, dim2=2))
        mask = ~torch.logical_or(torch.isinf(permuted_original_data), torch.isnan(permuted_original_data))
        valid_ranks = ranks[mask]
        
        if not torch.is_tensor(valid_ranks):
            valid_ranks = torch.tensor(valid_ranks)
            
        # Calculate metrics
        results = {f"R{k}": float(torch.sum(valid_ranks < k) * 100 / len(valid_ranks)) for k in top_k}
        results["MedianR"] = float(torch.median(valid_ranks + 1))
        results["MeanR"] = float(np.mean(valid_ranks.numpy() + 1))
        results["Std_Rank"] = float(np.std(valid_ranks.numpy() + 1))
        results['MR'] = results["MedianR"]
        
        return results
    
    @staticmethod
    def tensor_video_to_text_sim(sim_tensor: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Process a 3D similarity tensor for video-to-text evaluation.
        
        Args:
            sim_tensor: 3D tensor of similarity scores
            
        Returns:
            torch.Tensor: Processed similarity matrix for video-to-text evaluation
        """
        if not torch.is_tensor(sim_tensor):
            sim_tensor = torch.tensor(sim_tensor)
            
        # Replace NaN values with -inf
        sim_tensor[sim_tensor != sim_tensor] = float('-inf')
        
        # Get maximum similarity score for each video across all text descriptions
        values, _ = torch.max(sim_tensor, dim=1, keepdim=True)
        
        return torch.squeeze(values).T
    
    def print_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """
        Print retrieval metrics in a formatted way.
        
        Args:
            metrics: Dictionary containing retrieval metrics
            prefix: Optional prefix to add to the output (e.g., "Text-to-Video:")
        """
        if self.logger is None:
            return
            
        r1 = metrics['R1']
        r5 = metrics['R5']
        r10 = metrics['R10']
        r50 = metrics.get('R50', 0.0)  # 有些指标可能没有R50
        mr = metrics['MR']
        meanr = metrics["MeanR"]
        
        # 直接使用一个完整的字符串，避免多行日志输出
        metrics_str = f"{prefix}R@1: {r1:.1f} - R@5: {r5:.1f} - R@10: {r10:.1f} - R@50: {r50:.1f} - Median R: {mr:.1f} - Mean R: {meanr:.1f}"
        self.logger.info(metrics_str)
    
    def update_best_metrics(self, t2v_metrics: Dict[str, float], v2t_metrics: Dict[str, float], 
                           t2v_r1: float, v2t_r1: float) -> Tuple[bool, float]:
        """
        Update the best metrics if the current ones are better.
        
        Args:
            t2v_metrics: Text-to-video metrics dictionary
            v2t_metrics: Video-to-text metrics dictionary
            t2v_r1: Text-to-video R@1 score
            v2t_r1: Video-to-text R@1 score
            
        Returns:
            tuple: (is_updated, mean_r1)
                - is_updated: Whether any metrics were updated
                - mean_r1: Current mean R@1 score
        """
        is_updated = False
        
        # 更新文本到视频检索的最佳分数
        # Update best text-to-video retrieval score
        if self.best_t2v_r1 <= t2v_r1:
            self.best_t2v_r1 = t2v_r1
            self.best_t2v_metrics = t2v_metrics.copy()
            self.best_mean_r1 = (self.best_t2v_r1 + self.best_v2t_r1) / 2
            is_updated = True
            
        # 更新视频到文本检索的最佳分数
        # Update best video-to-text retrieval score
        if self.best_v2t_r1 <= v2t_r1:
            self.best_v2t_r1 = v2t_r1
            self.best_v2t_metrics = v2t_metrics.copy()
            self.best_mean_r1 = (self.best_t2v_r1 + self.best_v2t_r1) / 2
            is_updated = True
            
        # Calculate current mean R@1
        mean_r1 = (t2v_r1 + v2t_r1) / 2
        
        return is_updated, mean_r1
    
    def log_current_metrics(self, t2v_metrics: Dict[str, float], v2t_metrics: Dict[str, float], 
                           mean_r1: float) -> None:
        """
        Log the current evaluation metrics.
        
        Args:
            t2v_metrics: Text-to-video metrics dictionary
            v2t_metrics: Video-to-text metrics dictionary
            mean_r1: Mean R@1 score
        """
        if self.logger is None:
            return
            
        self.logger.info(f"Mean R@1: {mean_r1:.4f}")
        self.logger.info("Text-to-Video Retrieval:")
        self.print_metrics(t2v_metrics, prefix="  ")
        
        self.logger.info("Video-to-Text Retrieval:")
        self.print_metrics(v2t_metrics, prefix="  ")
    
    def log_best_metrics(self) -> None:
        """
        Log the best metrics achieved so far.
        """
        if self.logger is None or self.best_t2v_metrics is None or self.best_v2t_metrics is None:
            return
            
        self.logger.info(f"Best Mean R@1: {self.best_mean_r1:.4f}")
        self.logger.info("Best Text-to-Video Retrieval:")
        self.print_metrics(self.best_t2v_metrics, prefix="  ")
        
        self.logger.info("Best Video-to-Text Retrieval:")
        self.print_metrics(self.best_v2t_metrics, prefix="  ")
    
    def get_best_metrics(self) -> Dict[str, Union[float, Dict[str, float], None]]:
        """
        Get a dictionary containing the best metrics.
        
        Returns:
            dict: Dictionary containing best retrieval metrics:
                - 'score': Best mean R@1 score
                - 'text_to_video': Best text-to-video metrics
                - 'video_to_text': Best video-to-text metrics
                - 't2v_r1': Best text-to-video R@1 score
                - 'v2t_r1': Best video-to-text R@1 score
        """
        return {
            'score': self.best_mean_r1,
            'text_to_video': self.best_t2v_metrics,
            'video_to_text': self.best_v2t_metrics,
            't2v_r1': self.best_t2v_r1,
            'v2t_r1': self.best_v2t_r1
        }


# For backward compatibility
def compute_metrics(similarity_matrix):
    """
    Compute retrieval metrics from a similarity matrix.
    
    Args:
        similarity_matrix: Matrix of similarity scores between queries and targets
        
    Returns:
        dict: Dictionary containing retrieval metrics
    """
    return RetrievalMetrics.compute_metrics(similarity_matrix)


def tensor_text_to_video_metrics(sim_tensor, top_k=[1, 5, 10, 50]):
    """
    Compute text-to-video retrieval metrics from a 3D similarity tensor.
    
    Args:
        sim_tensor: 3D tensor of similarity scores
        top_k: List of k values for Recall@k calculation
        
    Returns:
        dict: Dictionary containing retrieval metrics
    """
    return RetrievalMetrics.tensor_text_to_video_metrics(sim_tensor, top_k)


def tensor_video_to_text_sim(sim_tensor):
    """
    Process a 3D similarity tensor for video-to-text evaluation.
    
    Args:
        sim_tensor: 3D tensor of similarity scores
        
    Returns:
        torch.Tensor: Processed similarity matrix for video-to-text evaluation
    """
    return RetrievalMetrics.tensor_video_to_text_sim(sim_tensor)


def print_computed_metrics(metrics):
    """
    Print retrieval metrics in a formatted way.
    
    Args:
        metrics: Dictionary containing retrieval metrics
    """
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    r50 = metrics.get('R50', 0.0)
    mr = metrics['MR']
    meanr = metrics["MeanR"]
    
    print(f"R@1: {r1:.1f} - R@5: {r5:.1f} - R@10: {r10:.1f} - R@50: {r50:.1f} - Median R: {mr:.1f} - Mean R: {meanr:.1f}")