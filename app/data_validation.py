import pandas as pd

def validate_uploaded_data(df: pd.DataFrame, expected_cols: list) -> tuple[bool, str]:
    """
    Validate uploaded data frame against expected columns and types.
    
    Args:
        df: Pandas DataFrame to validate
        expected_cols: List of expected column names
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if df is None or df.empty:
        return False, "Uploaded file is empty or missing data."
    
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns. Expected: {expected_cols}. Detected columns: {list(df.columns)}"
    
    # Check for non-numeric data or fully empty columns
    for col in expected_cols:
        df_col = pd.to_numeric(df[col], errors='coerce')
        if df_col.isna().all():
            return False, f"Column '{col}' contains entirely non-numeric data or is completely empty."
            
    return True, "Data is valid."
