import os
import pytest
import pandas as pd
import numpy as np

SAMPLE_FILE = "app/sample_data/aq16_glycerol_binary_format1.csv"

def test_sample_file_exists():
    """Verify that the sample data file exists."""
    assert os.path.exists(SAMPLE_FILE), f"Sample file {SAMPLE_FILE} not found."

def test_sample_data_delimiter_and_columns():
    """Verify delimiter handling and expected columns."""
    df = pd.read_csv(SAMPLE_FILE)
    
    # Check if expected columns are present
    expected_cols = ["concentration", "dG", "dH", "TdS"]
    for col in expected_cols:
        assert col in df.columns, f"Expected column '{col}' missing from sample data."

def test_sample_data_numeric_parsing():
    """Verify that numeric columns parse correctly as numeric types."""
    df = pd.read_csv(SAMPLE_FILE)
    
    for col in ["concentration", "dG", "dH", "TdS"]:
        # Pandas should parse these as numeric (float64 or int64)
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column '{col}' is not numeric."

def test_sample_data_no_fully_empty_columns():
    """Verify that required columns are not fully empty."""
    df = pd.read_csv(SAMPLE_FILE)
    
    for col in ["concentration", "dG", "dH", "TdS"]:
        # Check that the column has at least one valid (non-null) numeric entry
        assert df[col].notna().sum() > 0, f"Column '{col}' is fully empty."
