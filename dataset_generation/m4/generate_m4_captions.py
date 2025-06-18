#!/usr/bin/env python
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
import base64
import textwrap
import pandas as pd
import json
import torch


sys.path.append("../../time_series_datasets")

from m4.m4_loader import get_m4_loader


def generate_caption(time_series_data, series_id, save_plot=False):
    """
    Generate a caption for the time-series using OpenAI API by uploading a plot image
    """
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found. Please specify API key in environment variable.")
        client = OpenAI(api_key=api_key)
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_series_data, marker='o', linestyle='-', markersize=0)
        plt.grid(True, alpha=0.3)
        
        temp_image_path = f"temp_plot_{series_id}.png"
        plt.savefig(temp_image_path)
        plt.close()
        
        with open(temp_image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in time series analysis."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Generate a detailed caption for the following time-series data:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}", "detail": "high"}}
                    ]}
                ],
                temperature=0.5,
                max_tokens=500,
                seed=42
            )
        
        if not save_plot and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        caption = response.choices[0].message.content
        print(f"Generated caption for series {series_id}")
        return caption
    
    except Exception as e:
        print(f"Error generating caption: {e}")
        return f"Error generating caption: {str(e)}"


def generate_captions_for_batch(series_batch, ids, save_plot=False):
    captions = {}
    series_data = {}
    
    for i in range(len(ids)):
        # Extract the time series data
        plot_data = extract_plot_data(series_batch[i])
        series_str = json.dumps(plot_data.tolist())
        series_data[ids[i]] = series_str
        
        # Generate caption
        caption = generate_caption(plot_data, ids[i], save_plot=save_plot)
        captions[ids[i]] = caption
    
    return captions, series_data


def extract_plot_data(series_tensor):
    non_zero_indices = torch.where(series_tensor != 0)[0]
    if len(non_zero_indices) > 0:
        last_non_zero = non_zero_indices[-1]
        tensor = series_tensor[:last_non_zero + 1]
    else:
        tensor = series_tensor
    
    return tensor.detach().cpu().contiguous().numpy()


START_ID = None

try:
    print("Loading M4 Monthly data...")
    data_loader = get_m4_loader("Monthly", split="all", batch_size=4, shuffle=False)
    
    csv_file = "m4_captions_series.csv"
    
    skip_mode = START_ID is not None
    
    for series_batch, ids in data_loader:
        print(f"Batch shape: {series_batch.shape}")
        print(f"Series IDs: {ids}")
        
        if skip_mode and START_ID not in ids:
            print(f"Skipping batch, waiting for ID: {START_ID}")
            continue
        elif skip_mode:
            skip_mode = False
        
        print("\nGenerating captions for time series data...")
        captions, series_data = generate_captions_for_batch(
            series_batch,
            ids,
            save_plot=True
        )
        
        batch_df = pd.DataFrame([{
            "Caption": captions[sid],
            "Series": series_data[sid]  
        } for sid in ids])
        
        batch_df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
        
        # Remove the break to process all data
        break


except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()


