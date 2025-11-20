#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
TimeSeriesExam module for OpenTSLM.

This module provides dataset loaders for the TimeSeriesExam benchmark datasets.
"""

from .TimeSeriesExam1QADataset import TimeSeriesExam1QADataset

__all__ = ["TimeSeriesExam1QADataset"]
