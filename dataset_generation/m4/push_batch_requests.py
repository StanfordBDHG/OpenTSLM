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


def main():
    client = OpenAI()

    file = upload_file(client, "m4_caption_requests.jsonl")
    if file is None:
        print("Failed to upload file. Exiting.")
        return
    
    print(file)
    batch = create_batch(client, file.id)
    if batch is None:
        print("Failed to create batch. Exiting.")
        return
    
    print(batch)


if __name__ == "__main__":
    main()