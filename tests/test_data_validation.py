import pytest
import pandas as pd
import numpy as np
from app.data_validation import validate_uploaded_data

def test_missing_concentration_column():
    """Verify that a missing concentration column is rejected."""
    df = pd.DataFrame({"dG": [1.0, 2.0], "TdS": [0.5, 0.6]})
    is_valid, msg = validate_uploaded_data(df, expected_cols=["concentration", "dG"])
    assert not is_valid
    assert "Missing required columns" in msg
    assert "concentration" in msg

def test_missing_dg_column():
    """Verify that a missing potential (dG/dH/TdS) column is rejected."""
    df = pd.DataFrame({"concentration": [0.1, 0.2]})
    is_valid, msg = validate_uploaded_data(df, expected_cols=["concentration", "dG"])
    assert not is_valid
    assert "Missing required columns" in msg
    assert "dG" in msg

def test_nonnumeric_data():
    """Verify that non-numeric data is rejected."""
    df = pd.DataFrame({"concentration": ["0.1", "0.2"], "dG": ["abc", "def"]})
    is_valid, msg = validate_uploaded_data(df, expected_cols=["concentration", "dG"])
    assert not is_valid
    assert "entirely non-numeric data" in msg
    assert "dG" in msg

def test_empty_file():
    """Verify that an empty dataframe is rejected."""
    df = pd.DataFrame()
    is_valid, msg = validate_uploaded_data(df, expected_cols=["concentration", "dG"])
    assert not is_valid
    assert "empty or missing data" in msg

def test_ternary_missing_conc2_or_conc3():
    """Verify that ternary validations require both conc2 and conc3."""
    # Missing conc3
    df = pd.DataFrame({"conc2": [0.1, 0.2], "dG": [1.0, 2.0]})
    is_valid, msg = validate_uploaded_data(df, expected_cols=["conc2", "conc3", "dG"])
    assert not is_valid
    assert "Missing required columns" in msg
    assert "conc3" in msg

def test_valid_data():
    """Verify that valid numeric data passes validation."""
    df = pd.DataFrame({"conc2": [0.1, 0.2], "conc3": [0.05, 0.05], "dG": [1.0, 2.0]})
    is_valid, msg = validate_uploaded_data(df, expected_cols=["conc2", "conc3", "dG"])
    assert is_valid
    assert "Data is valid" in msg
