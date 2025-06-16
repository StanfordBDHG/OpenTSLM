#!/usr/bin/env python
import os
import argparse
import json
import time
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

def retrieve_batch_status(client, batch_id):
    """Retrieve the status of a batch job"""
    try:
        batch = client.batches.retrieve(batch_id)
        return batch
    except Exception as e:
        print(f"Error retrieving batch status: {e}")
        return None

def download_batch_results(client, batch):
    """Download and parse batch results"""
    try:
        # Get the output file ID from the batch object
        output_file_id = batch.output_file_id
        if not output_file_id:
            print("No output file ID found in batch")
            return None
            
        # Download the file content
        content = client.files.content(output_file_id)
        content_str = content.read().decode('utf-8')
        
        # Parse the JSONL response
        results = []
        for line in content_str.strip().split('\n'):
            results.append(json.loads(line))
        
        return results
    except Exception as e:
        print(f"Error downloading batch results: {e}")
        return None

def get_batch_ids(client, limit = None):
    """Extract only the batch IDs from all batches stored in OpenAI"""
    try:
        batches = client.batches.list(limit=limit)
        return [batch.id for batch in batches.data]
    except Exception as e:
        print(f"Error listing batches: {e}")
        return []


def parse_batch_results(results):
    """Parse the custom_id and content with the time series description from batch results"""
    if not results:
        return []
    
    parsed_data = []
    for item in results:
        if 'custom_id' in item and 'response' in item and 'body' in item['response']:
            body = item['response']['body']
            if 'choices' in body and len(body['choices']) > 0:
                content = body['choices'][0]['message'].get('content')
                parsed_data.append({
                    'custom_id': item['custom_id'],
                    'content': content
                })
    
    return parsed_data


def merge_time_series_with_captions(captions, time_series_file):
    """Merge time series data with captions into a single CSV"""
    try:
        # Read the time series data
        time_series_df = pd.read_csv(time_series_file)
        
        # Convert captions to DataFrame
        captions_df = pd.DataFrame(captions)
        captions_df = captions_df.rename(columns={'custom_id': 'custom_id', 'content': 'caption'})
        
        # Merge the two DataFrames on custom_id
        merged_df = pd.merge(time_series_df, captions_df, on='custom_id', how='left')
        
        return merged_df
    except Exception as e:
        print(f"Error merging data: {e}")
        return pd.DataFrame(captions)



def main():
    client = OpenAI()
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process batch requests and merge with time series data')
    parser.add_argument('--requests', help='Path to the requests JSONL file')
    parser.add_argument('--time-series', default='m4_series.csv', help='Path to the time series CSV file')
    args = parser.parse_args()
    
    # List batches
    batch_ids = get_batch_ids(client, limit=10)
    captions = []
    for batch_id in batch_ids:
        batch = retrieve_batch_status(client, batch_id)
        results = download_batch_results(client, batch)
        
        parsed_data = parse_batch_results(results)
        captions.extend(parsed_data)
    
    # If we have time series data, merge it with the captions
    if os.path.exists(args.time_series):
        merged_df = merge_time_series_with_captions(captions, args.time_series)
        output_filename = "m4_series.csv"  # Overwrite the original file with merged data
        merged_df.to_csv(output_filename, index=False)
        print(f"Results merged with time series data and saved to {output_filename}")
    else:
        # Just save the captions if no time series data is available
        df = pd.DataFrame(captions)
        df = df.rename(columns={'custom_id': 'custom_id', 'content': 'caption'})
        output_filename = "m4_captions.csv"
        df.to_csv(output_filename, index=False)
        print(f"Results saved to {output_filename}")
        print(f"Warning: Time series file {args.time_series} not found. Only captions were saved.")



if __name__ == "__main__":
    main()



#
# (venv) (base) maxrosenblattl@9028007b-fead-4c10-b482-3ad4d8975c3b m4 % python3 process_batch_requests.py
# FileObject(id='file-757JtesN29hmNHpMabrqpf', bytes=192844, created_at=1749855551, filename='m4_caption_requests_20250614_004332.jsonl', object='file', purpose='batch', status='processed', expires_at=None, status_details=None)
# Batch(id='batch_684cad40ea048190ac5db0d3b0b7399f', completion_window='24h', created_at=1749855552, endpoint='/v1/chat/completions', input_file_id='file-757JtesN29hmNHpMabrqpf', object='batch', status='validating', cancelled_at=None, cancelling_at=None, completed_at=None, error_file_id=None, errors=None, expired_at=None, expires_at=1749941952, failed_at=None, finalizing_at=None, in_progress_at=None, metadata=None, output_file_id=None, request_counts=BatchRequestCounts(completed=0, failed=0, total=0))