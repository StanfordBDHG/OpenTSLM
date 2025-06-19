#!/usr/bin/env python
import os
import argparse
import json
import time
import glob
import pandas as pd
from openai import OpenAI


def upload_file(client, file_path):
    """Upload a file to OpenAI for batch processing"""
    try:
        with open(file_path, "rb") as file:
            response = client.files.create(
                file=file,
                purpose="batch"
            )
        return response
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None

def create_batch(client, file_id):
    """Create a batch job using the uploaded file"""
    try:
        batch = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        return batch
    except Exception as e:
        print(f"Error creating batch: {e}")
        return None


def main():
    client = OpenAI()

    # Find all JSONL files that start with "m4_"
    jsonl_files = glob.glob("m4_*caption_requests*.jsonl")
    
    if not jsonl_files:
        print("No m4_*caption_requests*.jsonl files found. Exiting.")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to process:")
    for file_path in jsonl_files:
        print(f"  - {file_path}")
    
    # Confirm with user before proceeding
    confirm = input(f"Are you sure you want to upload {len(jsonl_files)} files to OpenAI for batch processing? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled by user.")
        return


    # Process each file
    for file_path in jsonl_files:
        print(f"\nProcessing {file_path}...")
        
        # Upload file
        file = upload_file(client, file_path)
        if file is None:
            print(f"Failed to upload {file_path}. Skipping to next file.")
            continue
        
        print(f"File uploaded successfully: {file.id}")
        
        # Create batch
        batch = create_batch(client, file.id)
        if batch is None:
            print(f"Failed to create batch for {file_path}. Skipping to next file.")
            continue
        
        print(f"Batch created successfully: {batch.id}")
        print(f"Status: {batch.status}")
        
        # Add a small delay between API calls to avoid rate limits
        time.sleep(1)


if __name__ == "__main__":
    main()