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

def main():
    client = OpenAI()
    # file = upload_file(client, "m4_caption_requests_20250614_004332.jsonl")
    # print(file)
    # batch = create_batch(client, file.id)
    # print(batch)

    batch = retrieve_batch_status(client, "batch_684cad40ea048190ac5db0d3b0b7399f")
    print(batch)
    results = download_batch_results(client, batch)
    print(results)

if __name__ == "__main__":
    main()



#
# (venv) (base) maxrosenblattl@9028007b-fead-4c10-b482-3ad4d8975c3b m4 % python3 process_batch_requests.py
# FileObject(id='file-757JtesN29hmNHpMabrqpf', bytes=192844, created_at=1749855551, filename='m4_caption_requests_20250614_004332.jsonl', object='file', purpose='batch', status='processed', expires_at=None, status_details=None)
# Batch(id='batch_684cad40ea048190ac5db0d3b0b7399f', completion_window='24h', created_at=1749855552, endpoint='/v1/chat/completions', input_file_id='file-757JtesN29hmNHpMabrqpf', object='batch', status='validating', cancelled_at=None, cancelling_at=None, completed_at=None, error_file_id=None, errors=None, expired_at=None, expires_at=1749941952, failed_at=None, finalizing_at=None, in_progress_at=None, metadata=None, output_file_id=None, request_counts=BatchRequestCounts(completed=0, failed=0, total=0))