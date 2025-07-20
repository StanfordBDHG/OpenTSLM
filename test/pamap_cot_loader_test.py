#!/usr/bin/env python3
"""
Test script for the PAMAP2 CoT loader.
"""

import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', "src"))

# Import and set up global logger with verbose mode
from logger import get_logger, set_global_verbose

class TestPAMAP2CoTLoader(unittest.TestCase):
    """
    Unit tests for the PAMAP2 CoT loader functions.
    """
    def setUp(self):
        # Set up global logger with verbose mode for detailed output
        set_global_verbose(True)
        self.logger = get_logger()
        
        from time_series_datasets.pamap2.pamap2_cot_loader import load_pamap2_cot_splits
        self.load_pamap2_cot_splits = load_pamap2_cot_splits
        
        self.logger.loading("Loading PAMAP2 CoT dataset splits...")
        self.train, self.val, self.test = self.load_pamap2_cot_splits()
        self.logger.success(f"Dataset loaded successfully: Train={len(self.train)}, Val={len(self.val)}, Test={len(self.test)}")

    def test_dataset_sizes(self):
        """Test that the datasets are non-empty and splits are correct."""
        self.logger.info("Testing dataset sizes...")
        self.assertGreater(len(self.train), 0)
        self.assertGreater(len(self.val), 0)
        self.assertGreater(len(self.test), 0)
        self.logger.success("Dataset size tests passed")

    def test_sample_keys(self):
        """Test that a sample contains all required keys."""
        self.logger.info("Testing sample keys...")
        sample = self.train[0]
        required_keys = {"x_axis", "y_axis", "z_axis", "label"}
        self.assertTrue(required_keys.issubset(sample.keys()))
        self.logger.success("Sample keys test passed")

    def test_axis_content(self):
        """Test that the axis data are lists and non-empty."""
        self.logger.info("Testing axis content...")
        sample = self.train[0]
        for axis in ["x_axis", "y_axis", "z_axis"]:
            self.assertIsInstance(sample[axis], list)
            self.assertGreater(len(sample[axis]), 0)
            self.logger.debug(f"{axis}: length={len(sample[axis])}")
        self.logger.success("Axis content tests passed")

    def test_label_is_string(self):
        """Test that the label is a string and non-empty."""
        self.logger.info("Testing label format...")
        sample = self.train[0]
        self.assertIsInstance(sample["label"], str)
        self.assertGreater(len(sample["label"]), 0)
        self.logger.success(f"Label test passed: '{sample['label']}'")

    def test_example_data(self):
        """Print example data to show what the dataset looks like."""
        sample = self.train[0]
        self.logger.info("="*80)
        self.logger.info("EXAMPLE PAMAP2 COT DATASET SAMPLE")
        self.logger.info("="*80)
        self.logger.info(f"Label: '{sample['label']}'")
        if 'rationale' in sample:
            self.logger.info(f"Rationale: '{sample['rationale']}'")
        for axis in ["x_axis", "y_axis", "z_axis"]:
            self.logger.info(f"{axis}: length={len(sample[axis])}, first 5: {sample[axis][:5]}")
        self.logger.info("="*80)

class TestPAMAP2CoTQADataset(unittest.TestCase):
    """
    Unit tests for the PAMAP2CoTQADataset class.
    """
    def setUp(self):
        # Set up global logger with verbose mode for detailed output
        set_global_verbose(True)
        self.logger = get_logger()
        
        from time_series_datasets.pamap2.PAMAP2CoTQADataset import PAMAP2CoTQADataset
        self.PAMAP2CoTQADataset = PAMAP2CoTQADataset
        
        self.logger.loading("Initializing PAMAP2CoTQADataset...")
        self.train_dataset = self.PAMAP2CoTQADataset(split="train", EOS_TOKEN="")
        self.val_dataset = self.PAMAP2CoTQADataset(split="validation", EOS_TOKEN="")
        self.test_dataset = self.PAMAP2CoTQADataset(split="test", EOS_TOKEN="")
        self.logger.success(f"Datasets initialized: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")

    def test_dataset_sizes(self):
        """Test that the datasets are non-empty and splits are correct."""
        self.logger.info("Testing QA dataset sizes...")
        self.assertGreater(len(self.train_dataset), 0)
        self.assertGreater(len(self.val_dataset), 0)
        self.assertGreater(len(self.test_dataset), 0)
        self.logger.success("QA dataset size tests passed")

    def test_sample_keys(self):
        """Test that a sample contains all required keys."""
        self.logger.info("Testing QA sample keys...")
        sample = self.train_dataset[0]
        required_keys = {"answer", "pre_prompt", "post_prompt", "time_series", "time_series_text"}
        self.assertTrue(required_keys.issubset(sample.keys()))
        self.logger.success("QA sample keys test passed")

    def test_answer_is_rationale(self):
        """Test that the answer is a string (rationale)."""
        self.logger.info("Testing answer format...")
        sample = self.train_dataset[0]
        self.assertIsInstance(sample["answer"], str)
        self.assertGreater(len(sample["answer"]), 0)
        self.logger.success(f"Answer test passed: length={len(sample['answer'])}")

    def test_time_series_content(self):
        """Test that the time series and text are present and valid."""
        self.logger.info("Testing time series content...")
        sample = self.train_dataset[0]
        self.assertIsInstance(sample["time_series"], list)
        self.assertIsInstance(sample["time_series_text"], list)
        self.assertGreater(len(sample["time_series"][0]), 0)
        self.assertIsInstance(sample["time_series_text"][0], str)
        self.logger.success(f"Time series test passed: {len(sample['time_series'])} series")

    def test_time_series_text_includes_mean_std(self):
        """Test that each time_series_text includes 'mean' and 'std', and both are followed by a number."""
        import re
        self.logger.info("Testing time series text format...")
        sample = self.train_dataset[0]
        for i, text in enumerate(sample['time_series_text']):
            self.logger.debug(f"Testing text {i}: {text[:100]}...")
            self.assertIn('mean', text)
            self.assertIn('std', text)
            # Allow for any whitespace after 'mean' and 'std'
            mean_match = re.search(r"mean\s+(-?\d+\.\d+)", text)
            if not mean_match:
                self.logger.error(f"DEBUG: {repr(text)}")
            self.assertIsNotNone(mean_match, f"No mean value found in: {repr(text)}")
            std_match = re.search(r"std\s+(-?\d+\.\d+)", text)
            if not std_match:
                self.logger.error(f"DEBUG: {repr(text)}")
            self.assertIsNotNone(std_match, f"No std value found in: {repr(text)}")
        self.logger.success("Time series text format tests passed")

    def test_example_data_QA(self):
        """Print example data for PAMAP2CoTQADataset, showing all time series and text."""
        sample = self.train_dataset[0]
        self.logger.info("="*80)
        self.logger.info("EXAMPLE PAMAP2CoTQADataset SAMPLE")
        self.logger.info("="*80)
        self.logger.info(f"Pre-prompt: '{sample['pre_prompt']}'")
        self.logger.info(f"Post-prompt: '{sample['post_prompt']}'")
        self.logger.info(f"Answer (rationale): '{sample['answer']}'")
        self.logger.info(f"Number of time series: {len(sample['time_series'])}")
        for i, (ts, ts_text) in enumerate(zip(sample['time_series'], sample['time_series_text'])):
            self.logger.info(f"Time series {i} text: '{ts_text}'")
            self.logger.info(f"Time series {i} length: {len(ts)}")
            self.logger.info(f"First 10 values: {ts[:10]}")
            self.logger.info(f"Last 10 values: {ts[-10:]}")
        self.logger.info("="*80)

if __name__ == "__main__":
    unittest.main() 