import pytest

def test_fh_crowding_import():
    """Verify that the fh_crowding package can be imported successfully."""
    try:
        import fh_crowding
    except ImportError as e:
        pytest.fail(f"Failed to import fh_crowding: {e}")

def test_app_imports():
    """Verify that the main app modules can be imported without launching Streamlit."""
    try:
        # We try importing the utility modules from the app instead of app.py 
        # because importing app.py directly might run the Streamlit script context.
        import app.export
        import app.session_io
        import app.styles
    except ImportError as e:
        pytest.fail(f"Failed to import app modules: {e}")
