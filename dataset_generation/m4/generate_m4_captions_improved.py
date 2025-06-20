#!/usr/bin/env python
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
import base64
import pandas as pd
import json
import torch
import time
from tqdm import tqdm
import io

sys.path.append("../../time_series_datasets")

from m4.m4_loader import get_m4_loader


def generate_caption_efficient(time_series_data, series_id, client, save_plot=False):
    """
    Generate a caption for the time-series using OpenAI API with efficient image handling
    """
    try:
        # Create plot in memory instead of saving to disk
        plt.figure(figsize=(10, 6))
        plt.plot(time_series_data, marker='o', linestyle='-', markersize=0)
        plt.grid(True, alpha=0.3)
        
        # Save to memory buffer instead of disk
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        
        # Get image data from buffer
        img_buffer.seek(0)
        image_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        # Make API request
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
        
        caption = response.choices[0].message.content
        return caption
    
    except Exception as e:
        print(f"Error generating caption for series {series_id}: {e}")
        return f"Error generating caption: {str(e)}"


def extract_plot_data(series_tensor):
    """Extract the actual time series data from the tensor"""
    non_zero_indices = torch.where(series_tensor != 0)[0]
    if len(non_zero_indices) > 0:
        last_non_zero = non_zero_indices[-1]
        tensor = series_tensor[:last_non_zero + 1]
    else:
        tensor = series_tensor
    
    return tensor.detach().cpu().contiguous().numpy()


def load_existing_captions(csv_file):
    """Load existing captions to resume from where we left off"""
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        return set(df['id'].tolist())
    return set()


def main():
    # Configuration
    FREQUENCY = "Weekly"  # Change this to your desired frequency
    START_ID = None  # Set this to resume from a specific ID
    CAPTIONS_FILE = f"m4_captions_{FREQUENCY}.csv"
    SERIES_FILE = f"m4_series_{FREQUENCY}.csv"
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable not found. Please set your API key.")
        return
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Load existing captions to avoid duplicates
    existing_ids = load_existing_captions(CAPTIONS_FILE)
    print(f"Found {len(existing_ids)} existing captions")
    
    try:
        print(f"Loading M4 {FREQUENCY} data...")
        data_loader = get_m4_loader(FREQUENCY, split="all", batch_size=1, shuffle=False)
        
        # Track progress
        total_processed = 0
        total_captions = 0
        all_captions = []
        all_series = []
        
        skip_mode = START_ID is not None
        
        for series_tensor, series_id in tqdm(data_loader, desc="Processing series"):
            series_id = series_id[0]  # Extract from batch
            series_tensor = series_tensor[0]  # Extract from batch
            
            # Skip if we're waiting for a specific ID
            if skip_mode and series_id != START_ID:
                continue
            elif skip_mode:
                skip_mode = False
            
            # Skip if already processed
            if f"series-{series_id}" in existing_ids:
                continue
            
            print(f"Processing series {series_id}")
            
            # Extract time series data
            plot_data = extract_plot_data(series_tensor)
            
            # Generate caption
            caption = generate_caption_efficient(plot_data, series_id, client)
            
            # Store results
            caption_data = {
                'id': f"series-{series_id}",
                'caption': caption
            }
            
            series_data = {
                'id': f"series-{series_id}",
                'series': json.dumps(plot_data.tolist())
            }
            
            # Save immediately to avoid losing progress
            caption_df = pd.DataFrame([caption_data])
            series_df = pd.DataFrame([series_data])
            
            caption_df.to_csv(CAPTIONS_FILE, mode='a', header=not os.path.exists(CAPTIONS_FILE), index=False)
            series_df.to_csv(SERIES_FILE, mode='a', header=not os.path.exists(SERIES_FILE), index=False)
            
            all_captions.append(caption_data)
            all_series.append(series_data)
            total_processed += 1
            total_captions += 1
            
            print(f"Generated caption: {caption[:100]}...")
            print(f"Total processed: {total_processed}")
            
            # Add small delay to avoid rate limits
            time.sleep(0.5)
            
            # Optional: limit for testing
            # if total_processed >= 5:  # Uncomment to limit processing
            #     break
        
        print(f"\nProcessing complete!")
        print(f"Total series processed: {total_processed}")
        print(f"Total captions generated: {total_captions}")
        print(f"Captions saved to: {CAPTIONS_FILE}")
        print(f"Series data saved to: {SERIES_FILE}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 