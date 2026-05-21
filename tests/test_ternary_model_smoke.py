import pytest
import numpy as np
from fh_crowding import Protein, CosoluteMixture, TernaryCrowdingModel

def test_ternary_model_smoke():
    """Verify that the TernaryCrowdingModel can solve equilibrium and generate results."""
    # Create basic inputs
    protein = Protein(SASA=419.0)
    cosolutes = CosoluteMixture(
        nu2=1.0, nu3=1.0,
        chi12=0.1, chi13=0.1, chi23=0.0,
        chiTS12=-0.05, chiTS13=-0.05, chiTS23=0.0,
        phi2_max=0.1, phi3_max=0.1,  # Keep grid extremely small
        dphi2=0.05, dphi3=0.05       # Coarse step
    )
    
    # Initialize model
    model = TernaryCrowdingModel(
        protein=protein,
        cosolutes=cosolutes,
        eps2=0.0, eps3=0.0, eps23=0.0,
        epsTS2=0.0, epsTS3=0.0, epsTS23=0.0,
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
    expected_cols = ["phi2", "phi3", "molar2", "molar3", "ddG", "ddH", "TddS"]
    for col in expected_cols:
        assert col in df.columns, f"Expected column '{col}' missing from ternary model output."
        
    # Verify values are finite
    for col in ["ddG", "ddH", "TddS"]:
        assert np.all(np.isfinite(df[col].values)), f"Column '{col}' contains non-finite values."
