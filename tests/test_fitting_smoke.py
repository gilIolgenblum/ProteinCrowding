import pytest
import numpy as np
from fh_crowding import Protein, Cosolute, BinaryCrowdingModel

@pytest.fixture
def base_binary_model():
    """Provides a basic binary model for fitting tests."""
    protein = Protein(SASA=419.0)
    cosolute = Cosolute(
        nu=1.0,
        chi=0.1,
        chiTS=-0.05,
        phiC_max=0.15,
        dphiC=0.05
    )
    return BinaryCrowdingModel(protein=protein, cosolute=cosolute, eps=0.0, epsTS=0.0, T=298.15)

def test_fit_eps_smoke(base_binary_model):
    """Verify that fit_eps executes without crashing on synthetic data."""
    model = base_binary_model
    
    # Synthetic experimental data
    exp_conc = np.array([0.05, 0.10, 0.15])
    exp_ddG = np.array([-1.0, -1.8, -2.5])
    
    # Execute fitting routine
    model.fit_eps(exp_conc, exp_ddG, concentration_type="phi")
    
    # Assert that the optimization result object exists and eps has been modified
    assert hasattr(model, 'res')
    assert model.res is not None
    assert isinstance(model.eps, float)

def test_fit_epsTS_smoke(base_binary_model):
    """Verify that fit_epsTS executes without crashing on synthetic data."""
    model = base_binary_model
    
    # Synthetic experimental data
    exp_conc = np.array([0.05, 0.10, 0.15])
    exp_ddH = np.array([-2.0, -3.8, -5.5])
    exp_TddS = np.array([-1.0, -2.0, -3.0])
    
    # Execute fitting routine
    model.fit_epsTS(exp_conc, exp_ddH, exp_TddS, concentration_type="phi")
    
    # Assert that the optimization result object exists and epsTS has been modified
    assert hasattr(model, 'resTS')
    assert model.resTS is not None
    assert isinstance(model.epsTS, float)
