#!/usr/bin/env python
import os
import json
import time
import glob
import pandas as pd
from openai import OpenAI
from tqdm import tqdm


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


def wait_for_batch_completion(client, batch_id, check_interval=30):
    """Wait for a batch to complete, checking status every check_interval seconds"""
    print(f"Waiting for batch {batch_id} to complete...")
    
    while True:
        batch = retrieve_batch_status(client, batch_id)
        if batch is None:
            print(f"Failed to retrieve status for batch {batch_id}")
            return False
        
        print(f"Batch {batch_id} status: {batch.status}")
        
        if batch.status == "completed":
            print(f"Batch {batch_id} completed successfully!")
            return True
        elif batch.status == "failed":
            print(f"Batch {batch_id} failed!")
            return False
        elif batch.status == "expired":
            print(f"Batch {batch_id} expired!")
            return False
        
        print(f"Waiting {check_interval} seconds before next status check...")
        time.sleep(check_interval)


def process_single_batch(client, file_path):
    """Process a single batch file: upload, wait for completion, and download results"""
    print(f"\n{'='*60}")
    print(f"Processing: {file_path}")
    print(f"{'='*60}")
    
    # Step 1: Check token limit
    print("Step 1: Checking token limit...")
    if not check_batch_token_limit(file_path):
        print(f"Batch {file_path} exceeds token limit. Skipping.")
        return None
    
    # Step 2: Upload file
    print("Step 2: Uploading file...")
    file = upload_file(client, file_path)
    if file is None:
        print(f"Failed to upload {file_path}. Skipping.")
        return None
    
    print(f"File uploaded successfully: {file.id}")
    
    # Step 3: Create batch
    print("Step 3: Creating batch...")
    batch = create_batch(client, file.id)
    if batch is None:
        print(f"Failed to create batch for {file_path}. Skipping.")
        return None
    
    print(f"Batch created successfully: {batch.id}")
    
    # Step 4: Wait for completion
    print("Step 4: Waiting for batch completion...")
    success = wait_for_batch_completion(client, batch.id)
    if not success:
        print(f"Batch {batch.id} did not complete successfully. Skipping.")
        return None
    
    # Step 5: Download results
    print("Step 5: Downloading results...")
    results = download_batch_results(client, batch.id)
    if results is None:
        print(f"Failed to download results for batch {batch.id}. Skipping.")
        return None
    
    # Step 6: Parse results
    print("Step 6: Parsing results...")
    parsed_data = parse_batch_results(results)
    print(f"Extracted {len(parsed_data)} captions from batch")
    
    return parsed_data


def extract_frequency_from_filename(filename):
    """Extract frequency (e.g., 'Weekly') from filename like 'm4_Weekly_caption_requests_1.jsonl'"""
    try:
        # Split by underscore and get the second part (after 'm4')
        parts = filename.split('_')
        if len(parts) >= 2:
            return parts[1]  # This should be 'Weekly', 'Monthly', etc.
        return "Unknown"
    except:
        print(f"Error extracting frequency from filename: {filename}")
        exit()


def estimate_tokens_in_batch(file_path):
    """
    Estimate the number of tokens in a batch file by counting characters and using a rough approximation.
    This is a conservative estimate since we can't easily count exact GPT-4 tokens without the API.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Count the number of requests in the file
        lines = content.strip().split('\n')
        num_requests = len(lines)
        
        # Estimate tokens per request
        # Each request contains:
        # - System message (~20 tokens)
        # - User message with image description (~50 tokens)
        # - Base64 encoded image (roughly 4 chars per token for base64)
        # - JSON structure overhead (~30 tokens)
        
        total_tokens = 0
        for line in lines:
            try:
                request = json.loads(line)
                # Count characters in the request
                request_str = json.dumps(request)
                char_count = len(request_str)
                
                # Rough estimate: 1 token ≈ 4 characters for English text
                # For base64 images, it's more like 1 token ≈ 4 characters
                estimated_tokens = char_count // 4
                total_tokens += estimated_tokens
                
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON line in {file_path}")
                continue
        
        return total_tokens, num_requests
        
    except Exception as e:
        print(f"Error estimating tokens in {file_path}: {e}")
        return None, 0


def check_batch_token_limit(file_path, max_tokens=90000):
    """
    Check if a batch file exceeds the token limit before uploading
    """
    estimated_tokens, num_requests = estimate_tokens_in_batch(file_path)
    
    if estimated_tokens is None:
        print(f"Could not estimate tokens for {file_path}. Proceeding with caution.")
        return True
    
    print(f"Estimated tokens in {file_path}: {estimated_tokens:,}")
    print(f"Number of requests: {num_requests}")
    print(f"Average tokens per request: {estimated_tokens // num_requests if num_requests > 0 else 0}")
    
    if estimated_tokens > max_tokens:
        print(f"WARNING: Batch exceeds {max_tokens:,} token limit!")
        print(f"Estimated tokens: {estimated_tokens:,}")
        print(f"Excess: {estimated_tokens - max_tokens:,} tokens")
        
        # Ask user if they want to proceed anyway
        proceed = input("Do you want to proceed anyway? (y/n): ")
        return proceed.lower() == 'y'
    
    print(f"✓ Batch is within token limit ({estimated_tokens:,} < {max_tokens:,})")
    return True


def main():
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable not found. Please set your API key.")
        return
    
    client = OpenAI()

    # Find all JSONL files that start with "m4_"
    jsonl_files = glob.glob("m4_*caption_requests*.jsonl")
    
    if not jsonl_files:
        print("No m4_*caption_requests*.jsonl files found. Exiting.")
        return
    
    # Sort files to ensure consistent processing order
    jsonl_files.sort()
    
    # Extract frequency from the first filename
    frequency = extract_frequency_from_filename(jsonl_files[0])
    print(f"Detected frequency: {frequency}")
    
    print(f"Found {len(jsonl_files)} JSONL files to process:")
    for i, file_path in enumerate(jsonl_files, 1):
        print(f"  {i}. {file_path}")
    
    # Confirm with user before proceeding
    confirm = input(f"\nAre you sure you want to process {len(jsonl_files)} files sequentially? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled by user.")
        return
    
    # Process each file sequentially
    all_captions = []
    
    for i, file_path in enumerate(jsonl_files, 1):
        print(f"\nProcessing file {i}/{len(jsonl_files)}: {file_path}")
        
        # Process the batch
        batch_captions = process_single_batch(client, file_path)
        
        if batch_captions:
            all_captions.extend(batch_captions)
            print(f"Successfully processed batch. Total captions so far: {len(all_captions)}")
        else:
            print(f"Failed to process batch for {file_path}")
        
        # Add a small delay between batches
        if i < len(jsonl_files):
            print("Waiting 5 seconds before next batch...")
            time.sleep(5)
    
    # Save all results
    if all_captions:
        print(f"\n{'='*60}")
        print(f"Processing complete! Total captions: {len(all_captions)}")
        print(f"{'='*60}")
        
        # Save to CSV with frequency in filename
        df = pd.DataFrame(all_captions)
        df = df.rename(columns={'custom_id': 'id', 'content': 'caption'})
        output_filename = f"m4_captions_{frequency}.csv"
        df.to_csv(output_filename, index=False)
        print(f"Results saved to {output_filename}")
        
        # Also save to JSON for backup with frequency in filename
        backup_filename = f"m4_captions_{frequency}_backup.json"
        with open(backup_filename, 'w') as f:
            json.dump(all_captions, f, indent=2)
        print(f"Backup saved to {backup_filename}")
    else:
        print("No captions were successfully processed.")


if __name__ == "__main__":
    main() 