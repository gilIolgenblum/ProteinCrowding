import pytest
import streamlit as st
from app.session_io import serialize_session_state, deserialize_session_file, validate_session_payload

@pytest.fixture
def mock_session_state(monkeypatch):
    """Provides a mocked Streamlit session state."""
    
    # We create a dictionary that mimics the Streamlit session state properties.
    class MockSessionState(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)

        def __setattr__(self, key, value):
            self[key] = value

    mock_state = MockSessionState({
        "SASA": 450.0,
        "use_T": True,
        "T_input": 310.15,
        "last_model_type": "Binary Crowding Model",
        "bin_cosolute_select": "Custom",
        "bin_nu": 1.5,
        "bin_chi": 0.2,
        "bin_chiTS": -0.01,
        "bin_eps_input": -1.0,
        "bin_epsts_input": -2.0,
        "bin_dphiC": 0.005,
        "bin_phiC_max": 0.2
    })
    
    # Monkeypatch st.session_state
    monkeypatch.setattr(st, "session_state", mock_state)
    return mock_state

def test_session_roundtrip(mock_session_state):
    """Verify that serialization and validation run without issues on session data."""
    # Serialize current mocked state
    payload_str = serialize_session_state()
    assert payload_str is not None
    assert isinstance(payload_str, str)
    
    # Deserialize the payload back into a dictionary
    restored_dict = deserialize_session_file(payload_str)
    assert isinstance(restored_dict, dict)
    
    # Validate the structure of the restored dictionary
    is_valid, msg = validate_session_payload(restored_dict)
    assert is_valid, f"Session validation failed: {msg}"
    
    # Check that numeric parameters round-trip accurately
    protein = restored_dict.get("protein", {})
    assert protein.get("SASA") == 450.0
    
    temp = restored_dict.get("temperature", {})
    assert temp.get("T") == 310.15
    
    bin_params = restored_dict.get("binary_params", {})
    assert bin_params.get("nu") == 1.5
    assert bin_params.get("chi") == 0.2
    assert bin_params.get("eps") == -1.0
