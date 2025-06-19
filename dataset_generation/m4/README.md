# M4 Time Series Caption Generation

This directory contains scripts for generating captions for M4 time series data using OpenAI's GPT-4 Vision API.

## Overview

The system provides two approaches for caption generation:

1. **Direct API Processing** (`generate_m4_captions.py`) - Real-time API calls for small datasets
2. **Batch API Processing** - Scalable batch processing for large datasets:
   - `generate_m4_captions_batch.py` - Prepare batch requests
   - `push_batch_requests.py` - Upload requests to OpenAI
   - `pull_batch_response.py` - Download and process results

## Prerequisites

1. **OpenAI API Key**: Set your API key as an environment variable:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

2. **Python Dependencies**: Install required packages:
   ```bash
   pip install openai pandas numpy matplotlib torch
   ```

3. **M4 Dataset**: Ensure the M4 dataset is available in the expected location.

## Usage

### Option 1: Direct API Processing (Small Datasets)

For processing small batches of time series data:

```bash
python generate_m4_captions.py
```

**Features:**
- Real-time processing
- Immediate feedback
- Suitable for testing and small datasets
- Saves results to `m4_captions_series.csv`

### Option 2: Batch API Processing (Large Datasets)

For processing large datasets efficiently:

#### Step 1: Prepare Batch Requests
```bash
python generate_m4_captions_batch.py
```

This script:
- Loads M4 data in batches of 2500 series
- Creates plots for each time series
- Encodes plots as base64 images
- Saves requests to JSONL files (`m4_Monthly_caption_requests_*.jsonl`)
- Saves series data to `m4_series.csv`

#### Step 2: Upload Requests to OpenAI
```bash
python push_batch_requests.py
```

This script:
- Finds all JSONL request files
- Uploads them to OpenAI for batch processing
- Creates batch jobs with 24-hour completion window
- Provides batch IDs for tracking

#### Step 3: Download Results
```bash
python pull_batch_response.py
```

This script:
- Retrieves all batch jobs from OpenAI
- Downloads completed batch results
- Parses captions from responses
- Saves results to `m4_captions.csv`

## Configuration

Edit `config.py` to customize:

- **API Settings**: Model, temperature, max tokens
- **Batch Processing**: Batch sizes, completion windows
- **Data Processing**: Frequencies, file naming
- **Plot Settings**: Figure sizes, styling

## File Structure

```
m4/
├── generate_m4_captions.py          # Direct API processing
├── generate_m4_captions_batch.py    # Batch request preparation
├── push_batch_requests.py           # Upload requests to OpenAI
├── pull_batch_response.py           # Download batch results
├── m4_utils.py                      # Shared utilities
├── config.py                        # Configuration settings
├── README.md                        # This file
├── m4_series.csv                    # Time series data (generated)
├── m4_captions.csv                  # Generated captions (generated)
└── m4_Monthly_caption_requests_*.jsonl  # Batch request files (generated)
```

## Output Files

### `m4_series.csv`
Contains time series data with columns:
- `id`: Series identifier (format: `series-{series_id}`)
- `series`: JSON-encoded time series data

### `m4_captions.csv`
Contains generated captions with columns:
- `id`: Series identifier (format: `series-{series_id}`)
- `caption`: Generated caption text

## Performance Considerations

### Memory Usage
- Each plot is ~70KB when base64 encoded
- 2500 series per batch ≈ 175MB per batch file
- Monitor memory usage for large datasets

### API Costs
- GPT-4 Vision API pricing applies
- Batch processing may be more cost-effective for large datasets
- Monitor usage in OpenAI dashboard

### Processing Time
- Direct API: ~1-2 seconds per series
- Batch API: 24-hour completion window
- Actual processing time depends on OpenAI queue

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

2. **Memory Issues**
   - Reduce batch size in `config.py`
   - Process fewer frequencies at once

3. **File Not Found Errors**
   - Ensure M4 dataset is in correct location
   - Check file paths in `m4_loader.py`

4. **Batch Processing Delays**
   - Check batch status in OpenAI dashboard
   - Wait for completion window (24 hours)

### Error Handling

All scripts include comprehensive error handling:
- API failures are logged and skipped
- Invalid data is filtered out
- Temporary files are cleaned up automatically

## Extending the System

### Adding New Frequencies
Edit `config.py`:
```python
DEFAULT_FREQUENCIES = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
```

### Customizing Caption Generation
Modify system messages in `config.py`:
```python
SYSTEM_MESSAGE = "You are an expert in time series analysis."
USER_MESSAGE_TEMPLATE = "Generate a detailed caption for the following time-series data:"
```

### Adding Data Validation
Extend validation functions in `m4_utils.py`:
```python
def validate_time_series_data(data):
    # Add custom validation logic
    pass
```

## Contributing

When contributing to this codebase:

1. Follow the existing code structure
2. Add appropriate error handling
3. Update configuration as needed
4. Test with small datasets first
5. Update documentation for new features 