#!/usr/bin/env python
"""
Configuration settings for M4 caption generation
"""
import os
from typing import List, Dict, Any


class M4Config:
    """Configuration class for M4 caption generation"""
    
    # API Settings
    OPENAI_MODEL = "gpt-4o"
    OPENAI_TEMPERATURE = 0.5
    OPENAI_MAX_TOKENS = 500
    OPENAI_SEED = 42
    
    # Batch Processing Settings
    MAX_BATCH_SIZE = 2500  # Maximum requests per batch (200MB limit)
    COMPLETION_WINDOW = "24h"
    
    # Data Processing Settings
    DEFAULT_FREQUENCIES = ["Monthly"]  # Can be expanded to all frequencies
    BATCH_SIZE = 2500
    
    # File Naming Conventions
    REQUEST_FILE_PREFIX = "m4"
    REQUEST_FILE_SUFFIX = "caption_requests"
    SERIES_DATA_FILE = "m4_series.csv"
    CAPTIONS_FILE = "m4_captions.csv"
    
    # Plot Settings
    DEFAULT_FIGSIZE = (12, 6)
    FIGSIZE_BY_LENGTH = {
        500: (12, 6),
        1500: (15, 5),
        float('inf'): (18, 4)
    }
    
    # System Messages
    SYSTEM_MESSAGE = "You are an expert in time series analysis."
    USER_MESSAGE_TEMPLATE = "Generate a detailed caption for the following time-series data:"
    
    @classmethod
    def get_figsize_for_length(cls, length: int) -> tuple:
        """Get appropriate figure size based on time series length"""
        for max_length, figsize in sorted(cls.FIGSIZE_BY_LENGTH.items()):
            if length < max_length:
                return figsize
        return cls.DEFAULT_FIGSIZE
    
    @classmethod
    def get_request_template(cls) -> Dict[str, Any]:
        """Get the base request template for OpenAI API"""
        return {
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": cls.OPENAI_MODEL,
                "temperature": cls.OPENAI_TEMPERATURE,
                "max_tokens": cls.OPENAI_MAX_TOKENS,
                "seed": cls.OPENAI_SEED
            }
        }
    
    @classmethod
    def get_messages_template(cls) -> List[Dict[str, str]]:
        """Get the messages template for OpenAI API"""
        return [
            {"role": "system", "content": cls.SYSTEM_MESSAGE},
            {"role": "user", "content": [
                {"type": "text", "text": cls.USER_MESSAGE_TEMPLATE},
                {"type": "image_url", "image_url": {"detail": "high"}}
            ]}
        ]
    
    @classmethod
    def validate_environment(cls) -> bool:
        """Validate that required environment variables are set"""
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY environment variable not found.")
            print("Please set your OpenAI API key:")
            print("export OPENAI_API_KEY='your-api-key-here'")
            return False
        return True


# Convenience functions
def get_config() -> M4Config:
    """Get the configuration instance"""
    return M4Config


def validate_config() -> bool:
    """Validate the configuration"""
    return M4Config.validate_environment() 