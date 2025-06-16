#!/usr/bin/env python
import os
import argparse
import json
import time
import pandas as pd
from openai import OpenAI

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


def main():
    client = OpenAI()

    # List batches
    batch_ids = get_batch_ids(client, limit=10)
    captions = []
    for batch_id in batch_ids:
        batch = retrieve_batch_status(client, batch_id)
        results = download_batch_results(client, batch)
        
        parsed_data = parse_batch_results(results)
        captions.extend(parsed_data)
    
    # Just save the captions
    df = pd.DataFrame(captions)
    df = df.rename(columns={'custom_id': 'id', 'content': 'caption'})
    output_filename = "m4_captions.csv"
    df.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")



if __name__ == "__main__":
    main()
