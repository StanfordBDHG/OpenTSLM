#!/usr/bin/env python
"""
Shared utilities for M4 caption generation
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def extract_plot_data(series_tensor):
    """
    Extract time series data from tensor, removing zero padding
    
    Args:
        series_tensor: PyTorch tensor containing time series data
        
    Returns:
        numpy.ndarray: Clean time series data without zero padding
    """
    non_zero_indices = torch.where(series_tensor != 0)[0]
    if len(non_zero_indices) > 0:
        last_non_zero = non_zero_indices[-1]
        tensor = series_tensor[:last_non_zero + 1]
    else:
        tensor = series_tensor
    
    return tensor.detach().cpu().contiguous().numpy()


def create_time_series_plot(time_series_data, series_id, save_path=None, figsize=None):
    """
    Create a matplotlib plot for time series data
    
    Args:
        time_series_data: numpy array of time series data
        series_id: identifier for the series
        save_path: optional path to save the plot
        figsize: optional tuple for figure size
        
    Returns:
        str: base64 encoded image data if save_path is None, otherwise None
    """
    import base64
    
    # Determine figure size based on data length
    if figsize is None:
        num_samples = len(time_series_data)
        if num_samples < 500:
            figsize = (12, 6)
        elif num_samples < 1500:
            figsize = (15, 5)
        else:
            figsize = (18, 4)
    
    plt.figure(figsize=figsize)
    plt.plot(time_series_data, marker='o', linestyle='-', markersize=0)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return None
    else:
        # Create temporary file for base64 encoding
        temp_path = f"temp_plot_{series_id}.png"
        plt.savefig(temp_path)
        plt.close()
        
        with open(temp_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return image_data


def validate_time_series_data(data):
    """
    Validate time series data for quality issues
    
    Args:
        data: numpy array of time series data
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    if data is None or len(data) == 0:
        return False
    
    # Check for all zeros
    if np.all(data == 0):
        return False
    
    # Check for all same values (no variation)
    if np.std(data) == 0:
        return False
    
    # Check for infinite or NaN values
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        return False
    
    return True


def get_available_frequencies():
    """
    Get list of available M4 frequencies
    
    Returns:
        list: Available frequency strings
    """
    return ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]


def check_api_key():
    """
    Check if OpenAI API key is available
    
    Returns:
        bool: True if API key is set, False otherwise
    """
    return os.environ.get("OPENAI_API_KEY") is not None 