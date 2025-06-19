#!/usr/bin/env python
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import base64
import textwrap
import pandas as pd
import json
import torch
from datetime import datetime

sys.path.append("../../time_series_datasets")

from m4.m4_loader import get_m4_loader


def prepare_caption_request(time_series_data, series_id, save_plot=False):
    """
    Prepare a request for the OpenAI batch API to generate a caption for the time-series
    by creating a plot image and encoding it as base64
    """
    try:
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
        
        temp_image_path = f"temp_plot_{series_id}.png"
        plt.savefig(temp_image_path)
        plt.close()
        
        with open(temp_image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        if not save_plot and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        # Create request object for batch API
        request = {
            "custom_id": f"series-{series_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are an expert in time series analysis."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Generate a detailed caption for the following time-series data:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}", "detail": "high"}}
                    ]}
                ],
                "temperature": 0.5,
                "max_tokens": 500,
                "seed": 42
            }
        }
        
        print(f"Prepared request for series {series_id}")
        return request
    
    except Exception as e:
        print(f"Error preparing request: {e}")
        return None


def prepare_requests_for_batch(series_batch, ids, save_plot=False):
    requests = []
    series_data = {}
    
    for i in range(len(ids)):
        plot_data = extract_plot_data(series_batch[i])
        series_str = json.dumps(plot_data.tolist())
        series_data[ids[i]] = series_str
        
        request = prepare_caption_request(plot_data, ids[i], save_plot=save_plot)
        if request:
            requests.append(request)
    
    return requests, series_data


def extract_plot_data(series_tensor):
    non_zero_indices = torch.where(series_tensor != 0)[0]
    if len(non_zero_indices) > 0:
        last_non_zero = non_zero_indices[-1]
        tensor = series_tensor[:last_non_zero + 1]
    else:
        tensor = series_tensor
    
    return tensor.detach().cpu().contiguous().numpy()


def save_requests_to_jsonl(requests, output_file):
    """Save requests to a JSONL file for batch processing"""
    with open(output_file, 'w') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')
    print(f"Saved {len(requests)} requests to {output_file}")


def save_series_to_csv(series_data, output_file):
    """Save time series data to a CSV file with custom_id and time series columns"""
    # Create a list of dictionaries with custom_id and time_series
    data_list = []
    for series_id, series_str in series_data.items():
        data_list.append({
            'id': f"series-{series_id}",
            'series': series_str
        })
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(data_list)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(series_data)} series data entries to {output_file}")


if __name__ == "__main__":
    START_ID = None
    BATCH_SIZE = 2500  # Maximum batch size is capped at 200MB / 50,000 requests -> 2500 TS ~ 175MB = 40 batch files in total 

    # Check for API key early
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable not found. Please set your API key.")
        sys.exit(1)

    try:
        skip_mode = START_ID is not None
        
        # List of .csv files to process
        # frequencies = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
        frequencies = ["Monthly"]
        
        all_series_data = {}
        
        for frequency in frequencies:
            print(f"Loading M4 {frequency} data...")
            data_loader = get_m4_loader(frequency, split="all", batch_size=BATCH_SIZE, shuffle=False)
            batch_id = 1

            for series_batch, ids in data_loader:
                requests_file = f"m4_{frequency}_caption_requests_{batch_id}.jsonl"

                print(f"Batch shape: {series_batch.shape}")
                print(f"Series IDs: {ids}")
                
                if skip_mode and START_ID not in ids:
                    print(f"Skipping batch, waiting for ID: {START_ID}")
                    continue
                elif skip_mode:
                    skip_mode = False
                
                print(f"\nPreparing requests for {frequency} time series data...")
                batch_requests, series_data = prepare_requests_for_batch(
                    series_batch,
                    ids,
                    save_plot=False
                )
                
                # Save requests directly to JSONL file for this batch
                save_requests_to_jsonl(batch_requests, requests_file)
                
                # Add series data to the combined dictionary
                all_series_data.update(series_data)

                batch_id += 1
                
                # Remove the break to process all data from each frequency
                break
        
        if all_series_data:
            combined_csv_file = "m4_series.csv"
            save_series_to_csv(all_series_data, combined_csv_file)
        
        print(f"\nTo process these requests, run the push_batch_requests.py script with:")
        print(f"python push_batch_requests.py")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
