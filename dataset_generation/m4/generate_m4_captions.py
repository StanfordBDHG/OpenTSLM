#!/usr/bin/env python
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from openai import OpenAI
import textwrap
import time

sys.path.append("../../time_series_datasets")

from m4.m4_loader import get_m4_loader


def generate_caption(time_series_data, series_id):
    """
    Generate a caption for the time-series using OpenAI API
    """
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        data_str = ", ".join(map(str, time_series_data))
        
        prompt = f"Generate a detailed description of the following time series data: {data_str}"
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in time series analysis. Provide concise, informative descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        caption = response.choices[0].message.content
        print(f"Generated caption for series {series_id}")
        return caption
    
    except Exception as e:
        print(f"Error generating caption: {e}")
        return f"Error generating caption: {str(e)}"


def generate_captions_for_batch(series_batch, ids):
    captions = {}
    caption_file = "m4_captions.txt"
    with open(caption_file, 'w') as f:
        f.write("Captions for M4 Monthly Time Series Data\n\n")
    
    # Generate captions for each series
    for i in range(len(ids)):
        caption = generate_caption(series_batch[i], ids[i])
        captions[ids[i]] = caption
        
        with open(caption_file, 'a') as f:
            f.write(f"Series ID: {ids[i]}\n")
            f.write(f"{caption}\n\n")
    
    print(f"Captions saved to '{caption_file}'")
    return captions, caption_file


def extract_plot_data(series_tensor):
    series_np = series_tensor.detach().numpy()
    series_np = np.nan_to_num(series_np)
    
    non_zero_indices = np.where(series_np != 0)[0]
    if len(non_zero_indices) > 0:
        last_non_zero = non_zero_indices[-1]
        plot_data = series_np[:last_non_zero + 1]
    else:
        plot_data = series_np
        
    return plot_data

def plot_time_series_batch(series_batch, ids, title="M4 Time Series Data", filename="m4_time_series_plot.png", captions=None):
    fig_height = 4 * len(ids) if captions else 3 * len(ids)
    plt.figure(figsize=(12, fig_height))
    
    for i in range(len(ids)):
        plt.subplot(len(ids), 1, i+1)
        
        plot_data = extract_plot_data(series_batch[i])
        
        plt.plot(plot_data, marker='o', linestyle='-', markersize=4)
        
        if captions and ids[i] in captions:
            caption = captions[ids[i]]
            
            wrapped_caption = textwrap.fill(caption, width=100)
            plt.figtext(0.1, 0.99 - (i/len(ids)), wrapped_caption, fontsize=8, wrap=True)
            plt.title(f'Series ID: {ids[i]} (caption available)')
        else:
            plt.title(f'Series ID: {ids[i]}')
            
        plt.xlabel('Time Step')
        plt.ylabel('Normalized Value')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(filename)
    print(f"Plot saved as '{filename}'")
    
    return plt.gcf()

try:
    print("Loading M4 Monthly data...")
    data_loader = get_m4_loader("Monthly", split="all", batch_size=4)
    
    for series_batch, ids in data_loader:
        print(f"Batch shape: {series_batch.shape}")
        print(f"Series IDs: {ids}")
        
        print("\nPlotting time series data")
        plot_time_series_batch(
            series_batch, 
            ids, 
            filename="m4_time_series_plot.png"
        )

        print("\nGenerating captions for time series data...")
        captions, caption_file = generate_captions_for_batch(
            series_batch,
            ids,
        )
        
        break

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")

