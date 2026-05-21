import pytest
import numpy as np
from fh_crowding import Protein, Cosolute, BinaryCrowdingModel

def test_binary_model_smoke():
    """Verify that the BinaryCrowdingModel can solve equilibrium and generate results."""
    # Create basic inputs
    protein = Protein(SASA=419.0)
    cosolute = Cosolute(
        nu=1.0,
        chi=0.1,
        chiTS=-0.05,
        phiC_max=0.15,
        dphiC=0.01  # Coarse grid for speed
    )
    
    # Initialize model
    model = BinaryCrowdingModel(
        protein=protein,
        cosolute=cosolute,
        eps=0.0,
        epsTS=0.0,
        T=298.15
    )
    
    # Run the core simulation method
    model.solve_equil()
    
    # Convert results to dataframe
    model.to_pandas()
    df = model.results
    
    # Verify dataframe exists and is not empty
    assert df is not None
    assert not df.empty
    
    # Check that expected core result columns exist
    expected_cols = ["phiC", "molar", "molal", "ddA", "ddE", "TddS"]
    for col in expected_cols:
        assert col in df.columns, f"Expected column '{col}' missing from model output."
        
    # Verify values are finite
    for col in ["ddA", "ddE", "TddS"]:
        assert np.all(np.isfinite(df[col].values)), f"Column '{col}' contains non-finite values."
