import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import fh_crowding
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import session_io
import export

st.set_page_config(page_title="FH Crowding Model", layout="wide")

def _display_and_export_plot(fig, filename, key):
    st.pyplot(fig)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download Plot (PNG)",
            data=export.fig_to_bytes(fig, "png"),
            file_name=f"{filename}.png",
            mime="image/png",
            key=f"{key}_png",
            use_container_width=True
        )
    with col2:
        st.download_button(
            "📥 Download Plot (SVG)",
            data=export.fig_to_bytes(fig, "svg"),
            file_name=f"{filename}.svg",
            mime="image/svg+xml",
            key=f"{key}_svg",
            use_container_width=True
        )

# ---------------------------------------------------------------------------
# Cosolute database — individual cosolutes
# Parameters from: Rösgen & Auton, JACS 2023 (DOI: 10.1021/jacs.3c08702), SI.
# nu: excluded volume; chi: FH non-ideal mixing; chiTS: entropy component of chi
# ---------------------------------------------------------------------------
COSOLUTE_DB = {
    "Custom":    {"nu":  1.000, "chi":  0.000, "chiTS":  0.000},
    "Glycerol":  {"nu":  3.950, "chi":  0.233, "chiTS": -0.480},
    "Glucose":   {"nu":  6.270, "chi":  0.317, "chiTS": -0.317},
    "Galactose": {"nu":  6.260, "chi":  0.350, "chiTS": -1.070},
    "Sorbitol":  {"nu":  6.710, "chi":  0.381, "chiTS": -0.290},
    "Trehalose": {"nu": 11.700, "chi":  0.433, "chiTS": -1.120},
    "Sucrose":   {"nu": 11.900, "chi":  0.452, "chiTS": -0.854},
    "Urea":      {"nu":  2.479, "chi":  0.610, "chiTS": -3.650},
    "TMAO":      {"nu":  3.980, "chi": -0.680, "chiTS": -5.708},
}

# ---------------------------------------------------------------------------
# Cosolute pair database — ternary presets
# Specifies which individual cosolutes make up the pair and their cross-
# interaction parameters chi23 / chiTS23.
# ---------------------------------------------------------------------------
COSOLUTE_PAIR_DB = {
    "Custom":               {"c2": "Custom",   "c3": "Custom",    "chi23":  0.000, "chiTS23":   0.000},
    "Glycerol + Trehalose": {"c2": "Glycerol", "c3": "Trehalose", "chi23":  4.500, "chiTS23":  10.000},
    "Urea + TMAO":          {"c2": "Urea",     "c3": "TMAO",      "chi23":  0.963, "chiTS23": -38.810},
}

COSOLUTE_NAMES      = list(COSOLUTE_DB.keys())
COSOLUTE_PAIR_NAMES = list(COSOLUTE_PAIR_DB.keys())


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def _sync_cosolute(select_key: str, nu_key: str, chi_key: str, chiTS_key: str) -> None:
    """Fill nu / chi / chiTS from a single-cosolute preset."""
    if select_key not in st.session_state:
        return
    params = COSOLUTE_DB[st.session_state[select_key]]
    st.session_state[nu_key]    = params["nu"]
    st.session_state[chi_key]   = params["chi"]
    st.session_state[chiTS_key] = params["chiTS"]


def _sync_cosolute_pair() -> None:
    """Fill all ternary parameters from a cosolute-pair preset."""
    if "tern_pair_select" not in st.session_state:
        return
    pair = COSOLUTE_PAIR_DB[st.session_state["tern_pair_select"]]
    c2 = COSOLUTE_DB[pair["c2"]]
    c3 = COSOLUTE_DB[pair["c3"]]
    st.session_state["nu2"]     = c2["nu"]
    st.session_state["chi12"]   = c2["chi"]
    st.session_state["chiTS12"] = c2["chiTS"]
    st.session_state["nu3"]     = c3["nu"]
    st.session_state["chi13"]   = c3["chi"]
    st.session_state["chiTS13"] = c3["chiTS"]
    st.session_state["chi23"]   = pair["chi23"]
    st.session_state["chiTS23"] = pair["chiTS23"]
    st.session_state["tern_cosolute2_select"] = pair["c2"]
    st.session_state["tern_cosolute3_select"] = pair["c3"]


def load_binary_sample_callback() -> None:
    """Load a binary sample dataset in a callback to avoid widget state mutation issues."""
    sample_sel = st.session_state.get("bin_sample_select", "None")
    if sample_sel == "None":
        st.session_state["exp_data_loaded"] = False
        st.session_state["exp_conc_G"] = None
        st.session_state["exp_conc_T"] = None
        st.session_state["exp_ddG"] = None
        st.session_state["exp_ddH"] = None
        st.session_state["exp_TddS"] = None
        st.session_state["raw_exp_ddG"] = None
        st.session_state["raw_err_ddG"] = None
        st.session_state["raw_exp_ddH"] = None
        st.session_state["raw_err_ddH"] = None
        st.session_state["raw_exp_TddS"] = None
        st.session_state["raw_err_TddS"] = None
        st.session_state["sample_load_error"] = None
        return
        
    sample_files = {
        "Glycerol (met16)": "met16_glycerol_binary_format1.csv",
        "TMAO (met16)": "met16_tmao_binary_format1.csv",
        "Trehalose (met16)": "met16_trehalose_binary_format1.csv",
        "Urea (met16)": "met16_urea_binary_format1.csv"
    }
    import os
    file_path = os.path.join("app", "sample_data", sample_files[sample_sel])
    try:
        df = pd.read_csv(file_path)
        df["concentration"] = pd.to_numeric(df["concentration"], errors='coerce')
        df = df.dropna(subset=["concentration"])
        
        st.session_state["exp_conc_G"] = df["concentration"].values
        st.session_state["exp_conc_T"] = df["concentration"].values
        st.session_state["exp_data_loaded"] = True
        st.session_state["uploaded_conc_unit"] = "molal"
        st.session_state["uploaded_energy_unit"] = "kJ/mol"
        
        # Automatically select and sync the cosolute preset in the sidebar
        cosolute_preset_map = {
            "Glycerol (met16)": ("Glycerol", 3.950, 0.233, -0.480),
            "TMAO (met16)": ("TMAO", 3.980, -0.680, -5.708),
            "Trehalose (met16)": ("Trehalose", 11.700, 0.433, -1.120),
            "Urea (met16)": ("Urea", 2.479, 0.610, -3.650)
        }
        if sample_sel in cosolute_preset_map:
            preset_name, nu_val, chi_val, chiTS_val = cosolute_preset_map[sample_sel]
            st.session_state["bin_cosolute_select"] = preset_name
            st.session_state["bin_nu"] = nu_val
            st.session_state["bin_chi"] = chi_val
            st.session_state["bin_chiTS"] = chiTS_val
        
        st.session_state["fit_success_msg"] = None
        st.session_state["fit_warning_msg"] = None
        st.session_state["sample_load_error"] = None
        
        if "dG" in df.columns:
            df["dG"] = pd.to_numeric(df["dG"], errors='coerce')
            st.session_state["raw_exp_ddG"] = df["dG"].values
            st.session_state["exp_ddG"] = df["dG"].values
        if "err_dG" in df.columns:
            df["err_dG"] = pd.to_numeric(df["err_dG"], errors='coerce')
            st.session_state["raw_err_ddG"] = df["err_dG"].values
            st.session_state["err_ddG"] = df["err_dG"].values
        else:
            st.session_state["raw_err_ddG"] = np.nan
            st.session_state["err_ddG"] = np.nan
            
        if "dH" in df.columns:
            df["dH"] = pd.to_numeric(df["dH"], errors='coerce')
            st.session_state["raw_exp_ddH"] = df["dH"].values
            st.session_state["exp_ddH"] = df["dH"].values
        if "err_dH" in df.columns:
            df["err_dH"] = pd.to_numeric(df["err_dH"], errors='coerce')
            st.session_state["raw_err_ddH"] = df["err_dH"].values
            st.session_state["err_ddH"] = df["err_dH"].values
        else:
            st.session_state["raw_err_ddH"] = np.nan
            st.session_state["err_ddH"] = np.nan
            
        if "TdS" in df.columns:
            df["TdS"] = pd.to_numeric(df["TdS"], errors='coerce')
            st.session_state["raw_exp_TddS"] = df["TdS"].values
            st.session_state["exp_TddS"] = df["TdS"].values
        if "err_TdS" in df.columns:
            df["err_TdS"] = pd.to_numeric(df["err_TdS"], errors='coerce')
            st.session_state["raw_err_TddS"] = df["err_TdS"].values
            st.session_state["err_TddS"] = df["err_TdS"].values
        else:
            st.session_state["raw_err_TddS"] = np.nan
            st.session_state["err_TddS"] = np.nan
            
    except Exception as ex:
        st.session_state["sample_load_error"] = f"Error loading sample file: {ex}"


def load_ternary_sample_callback() -> None:
    """Load a ternary sample dataset in a callback to avoid widget state mutation issues."""
    sample_sel = st.session_state.get("tern_sample_select", "None")
    if sample_sel == "None":
        st.session_state["exp_data_loaded"] = False
        st.session_state["exp_conc2"] = None
        st.session_state["exp_conc3"] = None
        st.session_state["exp_val_G"] = None
        st.session_state["exp_val_H"] = None
        st.session_state["exp_val_S"] = None
        st.session_state["raw_exp_val_G"] = None
        st.session_state["raw_exp_val_H"] = None
        st.session_state["raw_exp_val_S"] = None
        st.session_state["sample_load_error"] = None
        return
        
    sample_files = {
        "Glycerol + Trehalose (met16)": {
            "dG": "met16_gly_tre_ternary_format1_dG.csv",
            "dH": "met16_gly_tre_ternary_format1_dH.csv",
            "TdS": "met16_gly_tre_ternary_format1_TdS.csv"
        },
        "Urea + TMAO (aq16)": {
            "dG": "aq16_urea_tmao_ternary_format1_dG.csv",
            "dH": "aq16_urea_tmao_ternary_format1_dH.csv",
            "TdS": "aq16_urea_tmao_ternary_format1_TdS.csv"
        }
    }
    import os
    files = sample_files[sample_sel]
    try:
        df_G = pd.read_csv(os.path.join("app", "sample_data", files["dG"]))
        df_G["conc2"] = pd.to_numeric(df_G["conc2"], errors='coerce')
        df_G["conc3"] = pd.to_numeric(df_G["conc3"], errors='coerce')
        df_G["dG"] = pd.to_numeric(df_G["dG"], errors='coerce')
        df_G = df_G.dropna(subset=["conc2", "conc3", "dG"])
        
        st.session_state["exp_conc2"] = df_G["conc2"].values
        st.session_state["exp_conc3"] = df_G["conc3"].values
        st.session_state["raw_exp_val_G"] = df_G["dG"].values
        st.session_state["exp_val_G"] = df_G["dG"].values
        st.session_state["exp_data_loaded"] = True
        st.session_state["uploaded_conc_unit"] = "molal"
        st.session_state["uploaded_energy_unit"] = "kJ/mol"
        
        # Automatically select and sync the cosolute pair preset in the sidebar
        pair_preset_map = {
            "Glycerol + Trehalose (met16)": ("Glycerol + Trehalose", "Glycerol", "Trehalose", 3.950, 0.233, -0.480, 11.700, 0.433, -1.120, 4.500, 10.000),
            "Urea + TMAO (aq16)": ("Urea + TMAO", "Urea", "TMAO", 2.479, 0.610, -3.650, 3.980, -0.680, -5.708, 0.963, -38.810)
        }
        if sample_sel in pair_preset_map:
            pair_name, c2_name, c3_name, nu2_val, chi12_val, chiTS12_val, nu3_val, chi13_val, chiTS13_val, chi23_val, chiTS23_val = pair_preset_map[sample_sel]
            st.session_state["tern_pair_select"] = pair_name
            st.session_state["tern_cosolute2_select"] = c2_name
            st.session_state["tern_cosolute3_select"] = c3_name
            st.session_state["nu2"] = nu2_val
            st.session_state["chi12"] = chi12_val
            st.session_state["chiTS12"] = chiTS12_val
            st.session_state["nu3"] = nu3_val
            st.session_state["chi13"] = chi13_val
            st.session_state["chiTS13"] = chiTS13_val
            st.session_state["chi23"] = chi23_val
            st.session_state["chiTS23"] = chiTS23_val
            
        st.session_state["fit_success_msg"] = None
        st.session_state["fit_warning_msg"] = None
        st.session_state["sample_load_error"] = None
        
        try:
            df_H = pd.read_csv(os.path.join("app", "sample_data", files["dH"]))
            df_H["conc2"] = pd.to_numeric(df_H["conc2"], errors='coerce')
            df_H["conc3"] = pd.to_numeric(df_H["conc3"], errors='coerce')
            df_H["dH"] = pd.to_numeric(df_H["dH"], errors='coerce')
            df_H = df_H.dropna(subset=["conc2", "conc3", "dH"])
            st.session_state["raw_exp_val_H"] = df_H["dH"].values
            st.session_state["exp_val_H"] = df_H["dH"].values
        except Exception:
            pass
            
        try:
            df_S = pd.read_csv(os.path.join("app", "sample_data", files["TdS"]))
            df_S["conc2"] = pd.to_numeric(df_S["conc2"], errors='coerce')
            df_S["conc3"] = pd.to_numeric(df_S["conc3"], errors='coerce')
            df_S["TdS"] = pd.to_numeric(df_S["TdS"], errors='coerce')
            df_S = df_S.dropna(subset=["conc2", "conc3", "TdS"])
            st.session_state["raw_exp_val_S"] = df_S["TdS"].values
            st.session_state["exp_val_S"] = df_S["TdS"].values
        except Exception:
            pass
            
    except Exception as ex:
        st.session_state["sample_load_error"] = f"Error loading sample files: {ex}"


# ---------------------------------------------------------------------------
# Helper function to convert experimental concentrations to plotted/model units
# ---------------------------------------------------------------------------
def convert_exp_conc(exp_conc, from_type, to_type, model, is_ternary=False, cosolute_idx=2, exp_conc3=None):
    """
    Converts experimental concentrations to matching units.
    is_ternary: True if ternary model, False if binary.
    cosolute_idx: 2 or 3 (only used for ternary).
    exp_conc3: needed for ternary molal conversion.
    """
    exp_conc = np.array(exp_conc, dtype=float)
    if not is_ternary:
        nu = model.nu
        Vs = model.Vs
        # 1. Convert to volume fraction (phiC)
        if from_type == "phi":
            phi = exp_conc
        elif from_type == "molar":
            phi = exp_conc * nu * Vs
        elif from_type == "molal":
            K = exp_conc * 18 * nu * 1e-3
            phi = K / (1.0 + K)
        else:
            phi = exp_conc
            
        # 2. Convert from phiC to target
        if to_type in ["phi", "phiC"]:
            return phi
        elif to_type == "molar":
            return phi / (nu * Vs)
        elif to_type == "molal":
            return phi / (18 * (1.0 - phi) * nu) * 1000
        return phi
    else:
        # Ternary conversions
        nu2 = model.nu2
        nu3 = model.nu3
        Vs = model.Vs
        
        if from_type == "phi":
            phi2 = exp_conc if cosolute_idx == 2 else exp_conc3
            phi3 = exp_conc3 if cosolute_idx == 2 else exp_conc
            phi = phi2 if cosolute_idx == 2 else phi3
        elif from_type == "molar":
            phi = exp_conc * (nu2 if cosolute_idx == 2 else nu3) * Vs
        elif from_type == "molal":
            if exp_conc3 is None:
                exp_conc3 = np.zeros_like(exp_conc)
            x2 = exp_conc * 18 * nu2 * 1e-3
            x3 = exp_conc3 * 18 * nu3 * 1e-3
            phi2 = x2 / (1.0 + x2 + x3)
            phi3 = x3 / (1.0 + x2 + x3)
            phi = phi2 if cosolute_idx == 2 else phi3
        else:
            phi = exp_conc
            
        # Convert phi to target
        if to_type in ["phi", "phiC", "phi2", "phi3"]:
            return phi
        elif to_type == "molar":
            return phi / ((nu2 if cosolute_idx == 2 else nu3) * Vs)
        elif to_type == "molal":
            return phi / (18 * (1.0 - phi) * (nu2 if cosolute_idx == 2 else nu3)) * 1000
        return phi


# ---------------------------------------------------------------------------
# Helper function to read CSV with separator auto-detection (comma or tab)
# ---------------------------------------------------------------------------
def read_uploaded_csv(file_obj):
    pos = file_obj.tell()
    try:
        first_line = file_obj.readline()
        if isinstance(first_line, bytes):
            first_line = first_line.decode('utf-8', errors='ignore')
    except Exception:
        first_line = ""
    finally:
        file_obj.seek(pos)
    sep = '\t' if '\t' in first_line else ','
    return pd.read_csv(file_obj, sep=sep)


# ---------------------------------------------------------------------------
# Initialise session state defaults (only on first load)
# ---------------------------------------------------------------------------
_defaults = {
    # common configuration
    "SASA": 419.0,
    "use_T": True,
    "T_input": 298.15,

    # binary cosolute
    "bin_nu":    1.0,  "bin_chi":   0.1,  "bin_chiTS":  -0.05,
    "bin_dphiC": 0.001, "bin_phiC_max": 0.15,
    
    # ternary cosolute 2
    "nu2":       1.0,  "chi12":     0.1,  "chiTS12":    -0.05,
    "tern_dphi2": 0.001, "tern_phi2_max": 0.15,
    
    # ternary cosolute 3
    "nu3":       1.0,  "chi13":     0.1,  "chiTS13":    -0.05,
    "tern_dphi3": 0.001, "tern_phi3_max": 0.15,
    
    # ternary cross-interaction
    "chi23":     0.0,  "chiTS23":   0.0,
    "eps23":     0.0,  "epsTS23":   0.0,
    
    # experimental data keys
    "exp_conc_G": None, "exp_ddG": None, "err_ddG": None,
    "exp_conc_T": None, "exp_ddH": None, "exp_TddS": None, "err_ddH": None, "err_TddS": None,
    "exp_conc2": None, "exp_conc3": None, "exp_val_G": None, "exp_val_H": None, "exp_val_S": None,
    "raw_exp_ddG": None, "raw_err_ddG": None,
    "raw_exp_ddH": None, "raw_err_ddH": None,
    "raw_exp_TddS": None, "raw_err_TddS": None,
    "raw_exp_val_G": None, "raw_exp_val_H": None, "raw_exp_val_S": None,
    "exp_data_loaded": False,
    
    # widget key defaults to prevent state mismatch
    "bin_eps_input": 0.0, "bin_epsts_input": 0.0,
    "tern_eps2_input": 0.0, "tern_eps3_input": 0.0,
    "tern_epsts2_input": 0.0, "tern_epsts3_input": 0.0,
    
    # fitted parameters tracker (to persist fitted parameters to sidebar input values)
    "fitted_eps": None, "fitted_epsTS": None,
    "fitted_eps2": None, "fitted_eps3": None, "fitted_epsTS2": None, "fitted_epsTS3": None,
    "fit_updated": False,
    "fit_success_msg": None,
    "fit_warning_msg": None
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Programmatically update input widget values after a successful fit
# (Must be done before the widgets are instantiated in the current rerun)
if st.session_state.get("fit_updated"):
    if st.session_state.get("fitted_eps") is not None:
        st.session_state["bin_eps_input"] = st.session_state["fitted_eps"]
    if st.session_state.get("fitted_epsTS") is not None:
        st.session_state["bin_epsts_input"] = st.session_state["fitted_epsTS"]
    if st.session_state.get("fitted_eps2") is not None:
        st.session_state["tern_eps2_input"] = st.session_state["fitted_eps2"]
    if st.session_state.get("fitted_eps3") is not None:
        st.session_state["tern_eps3_input"] = st.session_state["fitted_eps3"]
    if st.session_state.get("fitted_epsTS2") is not None:
        st.session_state["tern_epsts2_input"] = st.session_state["fitted_epsTS2"]
    if st.session_state.get("fitted_epsTS3") is not None:
        st.session_state["tern_epsts3_input"] = st.session_state["fitted_epsTS3"]
    st.session_state["fit_updated"] = False


# ---------------------------------------------------------------------------
# Sidebar — common
# ---------------------------------------------------------------------------
st.title("FH Crowding Thermodynamic Model")

st.sidebar.header("Model Configuration")
model_type = st.sidebar.selectbox(
    "Model Type", ["Binary Crowding Model", "Ternary Crowding Model"]
)

# ---------------------------------------------------------------------------
# Sidebar — Save/Load Session
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
with st.sidebar.expander("💾 Save/Load Session", expanded=False):
    st.markdown("### Save Current Session")
    try:
        session_json = session_io.serialize_session_state()
        filename_prefix = "fh_binary" if model_type == "Binary Crowding Model" else "fh_ternary"
        st.download_button(
            "📥 Download Session JSON",
            data=session_json,
            file_name=f"{filename_prefix}_session.json",
            mime="application/json",
            key="download_session_btn",
            use_container_width=True
        )
    except Exception as ex:
        st.error(f"Error preparing session data: {ex}")
        
    st.markdown("---")
    st.markdown("### Load Session")
    uploaded_session = st.file_uploader("Upload JSON Session", type=["json"], key="session_uploader")
    if uploaded_session is not None:
        try:
            content = uploaded_session.read().decode("utf-8")
            payload = session_io.deserialize_session_file(content)
            is_valid, msg = session_io.validate_session_payload(payload)
            if not is_valid:
                st.error(f"Invalid session format: {msg}")
            else:
                st.success("Session file successfully validated!")
                # Simple summary preview
                st.markdown("**Session Summary:**")
                st.markdown(f"- Model: `{payload['model_type']}`")
                st.markdown(f"- Temp: `{payload['temperature']['T']} K`")
                st.markdown(f"- SASA: `{payload['protein']['SASA']}`")
                if st.button("Apply Restored Session", key="btn_apply_session", use_container_width=True):
                    session_io.apply_session_payload(payload)
                    st.success("Restored session state! Rerunning...")
                    st.rerun()
        except Exception as ex:
            st.error(f"Failed to parse session file: {ex}")
st.sidebar.markdown("---")


# Clear fit messages on sample selection change or model type change
if "last_bin_sample" not in st.session_state:
    st.session_state["last_bin_sample"] = "None"
if "last_tern_sample" not in st.session_state:
    st.session_state["last_tern_sample"] = "None"
if "last_model_type" not in st.session_state:
    st.session_state["last_model_type"] = model_type

current_bin_sample = st.session_state.get("bin_sample_select", "None")
if current_bin_sample != st.session_state["last_bin_sample"]:
    st.session_state["last_bin_sample"] = current_bin_sample
    st.session_state["fit_success_msg"] = None
    st.session_state["fit_warning_msg"] = None
    st.session_state["fitted_eps"] = None
    st.session_state["fitted_epsTS"] = None

current_tern_sample = st.session_state.get("tern_sample_select", "None")
if current_tern_sample != st.session_state["last_tern_sample"]:
    st.session_state["last_tern_sample"] = current_tern_sample
    st.session_state["fit_success_msg"] = None
    st.session_state["fit_warning_msg"] = None
    st.session_state["fitted_eps2"] = None
    st.session_state["fitted_eps3"] = None
    st.session_state["fitted_epsTS2"] = None
    st.session_state["fitted_epsTS3"] = None

if model_type != st.session_state["last_model_type"]:
    st.session_state["last_model_type"] = model_type
    st.session_state["fit_success_msg"] = None
    st.session_state["fit_warning_msg"] = None

st.sidebar.subheader("Protein")
SASA = st.sidebar.number_input("SASA", step=1.0, format="%.1f", key="SASA")
protein = fh_crowding.Protein(SASA=SASA)

st.sidebar.subheader("Temperature")
use_T = st.sidebar.checkbox("Include Temperature", key="use_T")
T = st.sidebar.number_input("Temperature (K)", step=0.1, format="%.2f", key="T_input") if use_T else 298.15


# ---------------------------------------------------------------------------
# Sidebar — Binary model
# ---------------------------------------------------------------------------
if model_type == "Binary Crowding Model":
    st.sidebar.subheader("Cosolute")

    st.sidebar.selectbox(
        "Select cosolute",
        COSOLUTE_NAMES,
        key="bin_cosolute_select",
        on_change=_sync_cosolute,
        kwargs={
            "select_key": "bin_cosolute_select",
            "nu_key":     "bin_nu",
            "chi_key":    "bin_chi",
            "chiTS_key":  "bin_chiTS",
        },
        help="Choosing a preset fills nu, chi, chiTS automatically. "
             "You can still edit the values below manually.",
    )

    nu    = st.sidebar.number_input("nu",    key="bin_nu",    step=0.01, format="%.4f")
    chi   = st.sidebar.number_input("chi",   key="bin_chi",   step=0.01, format="%.4f")
    chiTS = st.sidebar.number_input("chiTS", key="bin_chiTS", step=0.01, format="%.4f")

    st.sidebar.subheader("Soft Interaction Parameters")
    
    eps = st.sidebar.number_input("eps", step=0.01, format="%.4f", key="bin_eps_input")
    epsTS = st.sidebar.number_input("epsTS", step=0.01, format="%.4f", key="bin_epsts_input")

    st.sidebar.subheader("Concentration Grid")
    dphiC = st.sidebar.number_input(
        "dphiC (step size)",
        min_value=1e-5,
        max_value=0.05,
        step=0.0005,
        format="%.5f",
        key="bin_dphiC",
        help="Concentration grid step size. Smaller = finer grid, slower simulation. "
             "Package default: 0.0001. App default: 0.001 (10× faster).",
    )
    phiC_max = st.sidebar.number_input(
        "phiC_max (max concentration)",
        min_value=0.001,
        max_value=1.0,
        step=0.01,
        format="%.3f",
        key="bin_phiC_max",
        help="Maximum volume fraction of the grid. Default: 0.15.",
    )

    cosolute = fh_crowding.Cosolute(nu=nu, chi=chi, chiTS=chiTS)
    model = fh_crowding.BinaryCrowdingModel(
        protein=protein, cosolute=cosolute, eps=eps, epsTS=epsTS,
        dphiC=dphiC, phiC_max=phiC_max, T=T,
    )


# ---------------------------------------------------------------------------
# Sidebar — Ternary model
# ---------------------------------------------------------------------------
else:
    # --- Cosolute pair preset (top-level shortcut) ---
    st.sidebar.subheader("Cosolute Pair Preset")
    st.sidebar.selectbox(
        "Select cosolute pair",
        COSOLUTE_PAIR_NAMES,
        key="tern_pair_select",
        on_change=_sync_cosolute_pair,
        help="Selecting a pair fills all cosolute and cross-interaction parameters at once. "
             "You can still fine-tune individual values below.",
    )

    st.sidebar.markdown("---")

    # --- Cosolute 2 ---
    st.sidebar.subheader("Cosolute 2")
    st.sidebar.selectbox(
        "Select cosolute 2",
        COSOLUTE_NAMES,
        key="tern_cosolute2_select",
        on_change=_sync_cosolute,
        kwargs={
            "select_key": "tern_cosolute2_select",
            "nu_key":     "nu2",
            "chi_key":    "chi12",
            "chiTS_key":  "chiTS12",
        },
        help="Overrides cosolute-2 parameters only.",
    )
    nu2     = st.sidebar.number_input("nu2",     key="nu2",     step=0.01, format="%.4f")
    chi12   = st.sidebar.number_input("chi12",   key="chi12",   step=0.01, format="%.4f")
    chiTS12 = st.sidebar.number_input("chiTS12", key="chiTS12", step=0.01, format="%.4f")
    
    eps2    = st.sidebar.number_input("eps2",    step=0.01, format="%.4f", key="tern_eps2_input")
    epsTS2  = st.sidebar.number_input("epsTS2",  step=0.01, format="%.4f", key="tern_epsts2_input")

    # --- Cosolute 3 ---
    st.sidebar.subheader("Cosolute 3")
    st.sidebar.selectbox(
        "Select cosolute 3",
        COSOLUTE_NAMES,
        key="tern_cosolute3_select",
        on_change=_sync_cosolute,
        kwargs={
            "select_key": "tern_cosolute3_select",
            "nu_key":     "nu3",
            "chi_key":    "chi13",
            "chiTS_key":  "chiTS13",
        },
        help="Overrides cosolute-3 parameters only.",
    )
    nu3     = st.sidebar.number_input("nu3",     key="nu3",     step=0.01, format="%.4f")
    chi13   = st.sidebar.number_input("chi13",   key="chi13",   step=0.01, format="%.4f")
    chiTS13 = st.sidebar.number_input("chiTS13", key="chiTS13", step=0.01, format="%.4f")
    
    eps3    = st.sidebar.number_input("eps3",    step=0.01, format="%.4f", key="tern_eps3_input")
    epsTS3  = st.sidebar.number_input("epsTS3",  step=0.01, format="%.4f", key="tern_epsts3_input")

    # --- Cosolute–cosolute interactions ---
    st.sidebar.subheader("Cosolute–Cosolute Interactions")
    chi23   = st.sidebar.number_input("chi23",   key="chi23",   step=0.01, format="%.4f")
    chiTS23 = st.sidebar.number_input("chiTS23", key="chiTS23", step=0.01, format="%.4f")
    eps23   = st.sidebar.number_input("eps23",   step=0.01, format="%.4f", key="eps23")
    epsTS23 = st.sidebar.number_input("epsTS23", step=0.01, format="%.4f", key="epsTS23")

    # --- Concentration grid ---
    st.sidebar.subheader("Concentration Grid")
    dphi2 = st.sidebar.number_input(
        "dphi2 (step size for cosolute 2)",
        min_value=1e-5,
        max_value=0.05,
        step=0.0005,
        format="%.5f",
        key="tern_dphi2",
        help="Grid step for cosolute 2 axis. Package default: 0.0001. App default: 0.001.",
    )
    dphi3 = st.sidebar.number_input(
        "dphi3 (step size for cosolute 3)",
        min_value=1e-5,
        max_value=0.05,
        step=0.0005,
        format="%.5f",
        key="tern_dphi3",
        help="Grid step for cosolute 3 axis. Package default: 0.0001. App default: 0.001.",
    )
    phi2_max = st.sidebar.number_input(
        "phi2_max (max concentration 2)",
        min_value=0.001,
        max_value=1.0,
        step=0.01,
        format="%.3f",
        key="tern_phi2_max",
        help="Maximum volume fraction of cosolute 2. Default: 0.15.",
    )
    phi3_max = st.sidebar.number_input(
        "phi3_max (max concentration 3)",
        min_value=0.001,
        max_value=1.0,
        step=0.01,
        format="%.3f",
        key="tern_phi3_max",
        help="Maximum volume fraction of cosolute 3. Default: 0.15.",
    )

    cosolutes = fh_crowding.CosoluteMixture(
        nu2=nu2, nu3=nu3,
        chi12=chi12, chi13=chi13, chi23=chi23,
        chiTS12=chiTS12, chiTS13=chiTS13, chiTS23=chiTS23,
    )
    model = fh_crowding.TernaryCrowdingModel(
        protein=protein, cosolutes=cosolutes,
        eps2=eps2, eps3=eps3, eps23=eps23,
        epsTS2=epsTS2, epsTS3=epsTS3, epsTS23=epsTS23,
        dphi2=dphi2, dphi3=dphi3,
        phi2_max=phi2_max, phi3_max=phi3_max,
        T=T,
    )

# ---------------------------------------------------------------------------
# Automatic rerun if session has just been restored
# ---------------------------------------------------------------------------
if st.session_state.get("session_restored"):
    try:
        kw = {}
        if model_type == "Ternary Crowding Model":
            kw["print_msg"] = False
        model.solve_equil(**kw)
        model.to_pandas()
        st.session_state["solved_model"] = model
        st.session_state["solved_model_type"] = model_type
        st.session_state["session_restored"] = False
        st.session_state["fit_updated"] = False  # Reset widget state flags
    except Exception as ex:
        st.error(f"Error executing auto-solve after session restore: {ex}")
        st.session_state["session_restored"] = False

# ---------------------------------------------------------------------------
# Section 1: Experimental Data Upload (Expandable, at top of page)
# ---------------------------------------------------------------------------
st.header("📋 Experimental Data & Fitting Hub")
with st.expander("📊 Upload Experimental Data & Unit Settings (Optional)", expanded=False):
    st.markdown("Upload experimental files to fit interaction parameters and overlay on plots.")
    
    col_u1, col_u2 = st.columns([1, 3])
    with col_u1:
        uploaded_conc_unit = st.selectbox(
            "CSV Concentration Unit",
            ["phi", "molar", "molal"],
            key="uploaded_conc_unit",
            help="Choose the concentration unit used in your uploaded experimental CSV files."
        )
        uploaded_energy_unit = st.selectbox(
            "CSV Energy Unit",
            ["kcal/mol", "kJ/mol"],
            key="uploaded_energy_unit",
            help="Choose the energy unit used in your uploaded CSV files. They will be automatically converted to kJ/mol for consistent fitting."
        )
        energy_mult = 4.184 if uploaded_energy_unit == "kcal/mol" else 1.0
        
        # Dynamically scale active experimental energy arrays using energy_mult and the raw inputs
        for key in ["exp_ddG", "err_ddG", "exp_ddH", "err_ddH", "exp_TddS", "err_TddS", "exp_val_G", "exp_val_H", "exp_val_S"]:
            raw_key = "raw_" + key
            if raw_key in st.session_state and st.session_state[raw_key] is not None:
                st.session_state[key] = np.array(st.session_state[raw_key]) * energy_mult

    with col_u2:
        # Option A: Load Sample Dataset
        st.markdown("### 💡 Option A: Load Sample Dataset")
        if model_type == "Binary Crowding Model":
            sample_options = [
                "None",
                "Glycerol (met16)",
                "TMAO (met16)",
                "Trehalose (met16)",
                "Urea (met16)"
            ]
            sample_sel = st.selectbox(
                "Select a sample binary dataset",
                sample_options,
                key="bin_sample_select",
                on_change=load_binary_sample_callback
            )
            if st.session_state.get("sample_load_error"):
                st.error(st.session_state["sample_load_error"])
            elif sample_sel != "None":
                st.success(f"Sample binary dataset '{sample_sel}' loaded successfully! (in kJ/mol)")
        else:
            sample_options = [
                "None",
                "Glycerol + Trehalose (met16)",
                "Urea + TMAO (aq16)"
            ]
            sample_sel = st.selectbox(
                "Select a sample ternary dataset",
                sample_options,
                key="tern_sample_select",
                on_change=load_ternary_sample_callback
            )
            if st.session_state.get("sample_load_error"):
                st.error(st.session_state["sample_load_error"])
            elif sample_sel != "None":
                st.success(f"Sample ternary dataset '{sample_sel}' loaded successfully! (in kJ/mol)")

        st.markdown("### 📂 Option B: Upload Your Own CSV File(s)")
        if model_type == "Binary Crowding Model":
            upload_mode = st.radio(
                "Data Upload Format",
                ["Single CSV File (concentration, dG, dH, TdS)", "Separate CSV Files for dG, dH, TdS"],
                key="bin_upload_mode",
                horizontal=True
            )
            
            if upload_mode == "Single CSV File (concentration, dG, dH, TdS)":
                f = st.file_uploader("Upload Single CSV File", type=["csv"], key="bin_single_uploader")
                if f:
                    try:
                        df = read_uploaded_csv(f)
                        st.dataframe(df.head(), use_container_width=True)
                        if "concentration" not in df.columns:
                            st.error(f"Missing required column: 'concentration'\n\nExpected columns: 'concentration' and at least one of 'dG', 'dH', 'TdS'.\n\nDetected columns: {list(df.columns)}")
                        else:
                            available_data = [c for c in ["dG", "dH", "TdS"] if c in df.columns]
                            if not available_data:
                                st.error(f"Could not find any data columns to fit ('dG', 'dH', or 'TdS').\n\nDetected columns: {list(df.columns)}")
                            else:
                                df["concentration"] = pd.to_numeric(df["concentration"], errors='coerce')
                                df = df.dropna(subset=["concentration"])
                                
                                st.session_state["exp_conc_G"] = df["concentration"].values
                                st.session_state["exp_conc_T"] = df["concentration"].values
                                st.session_state["exp_data_loaded"] = True
                                
                                if "dG" in df.columns:
                                    df["dG"] = pd.to_numeric(df["dG"], errors='coerce')
                                    st.session_state["raw_exp_ddG"] = df["dG"].values
                                    st.session_state["exp_ddG"] = df["dG"].values * energy_mult
                                if "err_dG" in df.columns:
                                    df["err_dG"] = pd.to_numeric(df["err_dG"], errors='coerce')
                                    st.session_state["raw_err_ddG"] = df["err_dG"].values
                                    st.session_state["err_ddG"] = df["err_dG"].values * energy_mult
                                else:
                                    st.session_state["raw_err_ddG"] = np.nan
                                    st.session_state["err_ddG"] = np.nan
                                    
                                if "dH" in df.columns:
                                    df["dH"] = pd.to_numeric(df["dH"], errors='coerce')
                                    st.session_state["raw_exp_ddH"] = df["dH"].values
                                    st.session_state["exp_ddH"] = df["dH"].values * energy_mult
                                if "err_dH" in df.columns:
                                    df["err_dH"] = pd.to_numeric(df["err_dH"], errors='coerce')
                                    st.session_state["raw_err_ddH"] = df["err_dH"].values
                                    st.session_state["err_ddH"] = df["err_dH"].values * energy_mult
                                else:
                                    st.session_state["raw_err_ddH"] = np.nan
                                    st.session_state["err_ddH"] = np.nan
                                    
                                if "TdS" in df.columns:
                                    df["TdS"] = pd.to_numeric(df["TdS"], errors='coerce')
                                    st.session_state["raw_exp_TddS"] = df["TdS"].values
                                    st.session_state["exp_TddS"] = df["TdS"].values * energy_mult
                                if "err_TdS" in df.columns:
                                    df["err_TdS"] = pd.to_numeric(df["err_TdS"], errors='coerce')
                                    st.session_state["raw_err_TddS"] = df["err_TdS"].values
                                    st.session_state["err_TddS"] = df["err_TdS"].values * energy_mult
                                else:
                                    st.session_state["raw_err_TddS"] = np.nan
                                    st.session_state["err_TddS"] = np.nan
                                    
                                st.success("Experimental dataset loaded successfully!")
                    except Exception as ex:
                        st.error(f"Error reading CSV file: {ex}")
            else:
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    f_G = st.file_uploader("Upload dG CSV (concentration, dG)", type=["csv"], key="bin_g_uploader")
                with col_f2:
                    f_H = st.file_uploader("Upload dH CSV (concentration, dH)", type=["csv"], key="bin_h_uploader")
                with col_f3:
                    f_S = st.file_uploader("Upload TdS CSV (concentration, TdS)", type=["csv"], key="bin_s_uploader")
                
                if f_G:
                    try:
                        df = read_uploaded_csv(f_G)
                        if "concentration" not in df.columns or "dG" not in df.columns:
                            st.error(f"Missing required columns for dG. Expected: 'concentration', 'dG'.\n\nDetected columns: {list(df.columns)}")
                        else:
                            df["concentration"] = pd.to_numeric(df["concentration"], errors='coerce')
                            df["dG"] = pd.to_numeric(df["dG"], errors='coerce')
                            df = df.dropna(subset=["concentration", "dG"])
                            st.session_state["exp_conc_G"] = df["concentration"].values
                            st.session_state["raw_exp_ddG"] = df["dG"].values
                            st.session_state["exp_ddG"] = df["dG"].values * energy_mult
                            st.session_state["raw_err_ddG"] = df["err_dG"].values if "err_dG" in df.columns else np.nan
                            st.session_state["err_ddG"] = (df["err_dG"].values * energy_mult) if "err_dG" in df.columns else np.nan
                            st.session_state["exp_data_loaded"] = True
                            st.success("dG experimental data loaded!")
                    except Exception as ex:
                        st.error(f"Error reading dG CSV file: {ex}")
                        
                if f_H:
                    try:
                        df = read_uploaded_csv(f_H)
                        if "concentration" not in df.columns or "dH" not in df.columns:
                            st.error(f"Missing required columns for dH. Expected: 'concentration', 'dH'.\n\nDetected columns: {list(df.columns)}")
                        else:
                            df["concentration"] = pd.to_numeric(df["concentration"], errors='coerce')
                            df["dH"] = pd.to_numeric(df["dH"], errors='coerce')
                            df = df.dropna(subset=["concentration", "dH"])
                            st.session_state["exp_conc_T"] = df["concentration"].values
                            st.session_state["raw_exp_ddH"] = df["dH"].values
                            st.session_state["exp_ddH"] = df["dH"].values * energy_mult
                            st.session_state["raw_err_ddH"] = df["err_dH"].values if "err_dH" in df.columns else np.nan
                            st.session_state["err_ddH"] = (df["err_dH"].values * energy_mult) if "err_dH" in df.columns else np.nan
                            st.session_state["exp_data_loaded"] = True
                            st.success("dH experimental data loaded!")
                    except Exception as ex:
                        st.error(f"Error reading dH CSV file: {ex}")
                        
                if f_S:
                    try:
                        df = read_uploaded_csv(f_S)
                        if "concentration" not in df.columns or "TdS" not in df.columns:
                            st.error(f"Missing required columns for TdS. Expected: 'concentration', 'TdS'.\n\nDetected columns: {list(df.columns)}")
                        else:
                            df["concentration"] = pd.to_numeric(df["concentration"], errors='coerce')
                            df["TdS"] = pd.to_numeric(df["TdS"], errors='coerce')
                            df = df.dropna(subset=["concentration", "TdS"])
                            st.session_state["exp_conc_T"] = df["concentration"].values
                            st.session_state["raw_exp_TddS"] = df["TdS"].values
                            st.session_state["exp_TddS"] = df["TdS"].values * energy_mult
                            st.session_state["raw_err_TddS"] = df["err_TdS"].values if "err_TdS" in df.columns else np.nan
                            st.session_state["err_TddS"] = (df["err_TdS"].values * energy_mult) if "err_TdS" in df.columns else np.nan
                            st.session_state["exp_data_loaded"] = True
                            st.success("TdS experimental data loaded!")
                    except Exception as ex:
                        st.error(f"Error reading TdS CSV file: {ex}")
        else: # Ternary
            upload_mode = st.radio(
                "Data Upload Format",
                ["Columns (conc2, conc3, potential)", "2D Matrices (Matrix format)"],
                key="tern_upload_mode",
                horizontal=True
            )
            
            if upload_mode == "Columns (conc2, conc3, potential)":
                col_tf1, col_tf2, col_tf3 = st.columns(3)
                with col_tf1:
                    f_G = st.file_uploader("Upload dG CSV (conc2, conc3, dG)", type=["csv"], key="tern_g_uploader")
                with col_tf2:
                    f_H = st.file_uploader("Upload dH CSV (conc2, conc3, dH)", type=["csv"], key="tern_h_uploader")
                with col_tf3:
                    f_S = st.file_uploader("Upload TdS CSV (conc2, conc3, TdS)", type=["csv"], key="tern_s_uploader")
                
                if f_G:
                    try:
                        df = read_uploaded_csv(f_G)
                        if "conc2" not in df.columns or "conc3" not in df.columns or "dG" not in df.columns:
                            st.error(f"Missing required columns for ternary dG. Expected: 'conc2', 'conc3', 'dG'.\n\nDetected columns: {list(df.columns)}")
                        else:
                            df["conc2"] = pd.to_numeric(df["conc2"], errors='coerce')
                            df["conc3"] = pd.to_numeric(df["conc3"], errors='coerce')
                            df["dG"] = pd.to_numeric(df["dG"], errors='coerce')
                            df = df.dropna(subset=["conc2", "conc3", "dG"])
                            st.session_state["exp_conc2"] = df["conc2"].values
                            st.session_state["exp_conc3"] = df["conc3"].values
                            st.session_state["raw_exp_val_G"] = df["dG"].values
                            st.session_state["exp_val_G"] = df["dG"].values * energy_mult
                            st.session_state["exp_data_loaded"] = True
                            st.success("Ternary dG data loaded!")
                    except Exception as ex:
                        st.error(f"Error reading dG CSV file: {ex}")
                if f_H:
                    try:
                        df = read_uploaded_csv(f_H)
                        if "conc2" not in df.columns or "conc3" not in df.columns or "dH" not in df.columns:
                            st.error(f"Missing required columns for ternary dH. Expected: 'conc2', 'conc3', 'dH'.\n\nDetected columns: {list(df.columns)}")
                        else:
                            df["conc2"] = pd.to_numeric(df["conc2"], errors='coerce')
                            df["conc3"] = pd.to_numeric(df["conc3"], errors='coerce')
                            df["dH"] = pd.to_numeric(df["dH"], errors='coerce')
                            df = df.dropna(subset=["conc2", "conc3", "dH"])
                            st.session_state["exp_conc2"] = df["conc2"].values
                            st.session_state["exp_conc3"] = df["conc3"].values
                            st.session_state["raw_exp_val_H"] = df["dH"].values
                            st.session_state["exp_val_H"] = df["dH"].values * energy_mult
                            st.session_state["exp_data_loaded"] = True
                            st.success("Ternary dH data loaded!")
                    except Exception as ex:
                        st.error(f"Error reading dH CSV file: {ex}")
                if f_S:
                    try:
                        df = read_uploaded_csv(f_S)
                        if "conc2" not in df.columns or "conc3" not in df.columns or "TdS" not in df.columns:
                            st.error(f"Missing required columns for ternary TdS. Expected: 'conc2', 'conc3', 'TdS'.\n\nDetected columns: {list(df.columns)}")
                        else:
                            df["conc2"] = pd.to_numeric(df["conc2"], errors='coerce')
                            df["conc3"] = pd.to_numeric(df["conc3"], errors='coerce')
                            df["TdS"] = pd.to_numeric(df["TdS"], errors='coerce')
                            df = df.dropna(subset=["conc2", "conc3", "TdS"])
                            st.session_state["exp_conc2"] = df["conc2"].values
                            st.session_state["exp_conc3"] = df["conc3"].values
                            st.session_state["raw_exp_val_S"] = df["TdS"].values
                            st.session_state["exp_val_S"] = df["TdS"].values * energy_mult
                            st.session_state["exp_data_loaded"] = True
                            st.success("Ternary TdS data loaded!")
                    except Exception as ex:
                        st.error(f"Error reading TdS CSV file: {ex}")
            else:
                st.info("Matrix formats are typically fitted via columns. Please ensure CSV contains 'conc2', 'conc3' and data columns.")


# ---------------------------------------------------------------------------
# Section 2: Side-by-Side Simulation and Data Fitting Workspaces
# ---------------------------------------------------------------------------
col_sim, col_fit = st.columns(2)

with col_sim:
    st.subheader("⚡ Forward Simulation")
    st.markdown("Solve the equilibrium and calculate all thermodynamic quantities with current sidebar parameters.")
    
    # Run Simulation button
    if st.button("Run Simulation", key="run_sim_btn", use_container_width=True):
        progress_bar = st.progress(0, text="Solving equilibrium... 0%")

        def update_progress(frac: float) -> None:
            progress_bar.progress(frac, text=f"Solving equilibrium... {int(frac * 100)}%")

        try:
            # Clear old solved model while we run
            if "solved_model" in st.session_state:
                del st.session_state["solved_model"]
            
            kw = {"callback": update_progress}
            if model_type == "Ternary Crowding Model":
                kw["print_msg"] = False
            
            model.solve_equil(**kw)
            progress_bar.progress(1.0, text="Equilibrium solved!")

            model.to_pandas()
            st.success("Simulation complete!")
            
            # Store solved model object in session state so plots can persist and update dynamically!
            st.session_state["solved_model"] = model
            st.session_state["solved_model_type"] = model_type
            
        except Exception as e:
            st.error(f"Error during simulation: {e}")

    # Display solved model results and download if available
    if "solved_model" in st.session_state and st.session_state["solved_model_type"] == model_type:
        solved_model = st.session_state["solved_model"]
        
        st.markdown("### Export Simulated Quantities")
        csv = export.get_model_results_csv(solved_model)
        filename_prefix = "fh_crowding_binary" if model_type == "Binary Crowding Model" else "fh_crowding_ternary"
        st.download_button(
            "📥 Download Simulation Results (CSV)",
            data=csv,
            file_name=f"{filename_prefix}_model_results.csv",
            mime="text/csv",
            use_container_width=True
        )


with col_fit:
    st.subheader("🛠️ Parameter Fitting")
    st.markdown("Calculate soft interaction parameters (`eps` and `epsTS`) from your uploaded experimental dataset.")
    
    if not st.session_state["exp_data_loaded"]:
        st.info("💡 To fit interaction parameters, please upload experimental data files in the section at the top of the page.")
    else:
        st.markdown(f"**Loaded Experimental Data Unit:** `{uploaded_conc_unit}`")
        
        # Fit concentration type is determined by the uploaded CSV unit
        fit_conc_type = uploaded_conc_unit
        
        st.markdown("---")
        
        # Display persistent fit status messages if they exist
        if st.session_state.get("fit_success_msg"):
            st.success(st.session_state["fit_success_msg"])
        if st.session_state.get("fit_warning_msg"):
            st.warning(st.session_state["fit_warning_msg"])
            
        # Binary fitting controls
        if model_type == "Binary Crowding Model":
            col_b1, col_b2 = st.columns(2)
            
            # Button 1: Fit eps
            with col_b1:
                st.markdown("**Free Energy Soft Parameter**")
                fit_eps_btn = st.button("Fit eps (from dG)", key="btn_fit_eps", use_container_width=True)
                if fit_eps_btn:
                    if st.session_state.get("exp_ddG") is not None and st.session_state.get("exp_conc_G") is not None:
                        fit_progress = st.progress(0, text="Fitting eps...")
                        try:
                            # Filter NaNs
                            conc_G = np.array(st.session_state["exp_conc_G"])
                            ddG = np.array(st.session_state["exp_ddG"])
                            valid_idx = np.isfinite(conc_G) & np.isfinite(ddG)
                            if not np.any(valid_idx):
                                raise ValueError("No valid non-NaN experimental dG data points to fit.")
                            # Run fit
                            model.fit_eps(
                                conc_G[valid_idx],
                                ddG[valid_idx],
                                concentration_type=fit_conc_type
                            )
                            fit_progress.progress(0.5, text="Solving equilibrium with fitted eps...")
                            # Resolve model
                            model.solve_equil()
                            model.to_pandas()
                            fit_progress.progress(1.0, text="Fit & Simulation updated!")
                            
                            # Save state
                            st.session_state["fitted_eps"] = model.eps
                            st.session_state["fit_updated"] = True
                            st.session_state["solved_model"] = model
                            st.session_state["solved_model_type"] = model_type
                            
                            st.session_state["fit_success_msg"] = f"Successfully fitted eps: {model.eps:.4f}"
                            if hasattr(model, "res") and hasattr(model.res, "success") and not model.res.success:
                                msg = getattr(model.res, "message", "Optimizer did not converge.")
                                st.session_state["fit_warning_msg"] = f"⚠️ Optimization Warning: {msg}"
                            else:
                                st.session_state["fit_warning_msg"] = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Fitting error: {e}")
                    else:
                        st.error("Please upload experimental dG data first!")
                        
            # Button 2: Fit epsTS
            with col_b2:
                st.markdown("**Entropic Soft Parameter**")
                fit_epsts_btn = st.button("Fit epsTS (from dH, TdS)", key="btn_fit_epsts", use_container_width=True)
                if fit_epsts_btn:
                    if (st.session_state.get("exp_ddH") is not None and 
                        st.session_state.get("exp_TddS") is not None and 
                        st.session_state.get("exp_conc_T") is not None):
                        fit_progress = st.progress(0, text="Fitting epsTS...")
                        try:
                            # Filter NaNs
                            conc_T = np.array(st.session_state["exp_conc_T"])
                            ddH = np.array(st.session_state["exp_ddH"])
                            TddS = np.array(st.session_state["exp_TddS"])
                            valid_idx = np.isfinite(conc_T) & np.isfinite(ddH) & np.isfinite(TddS)
                            if not np.any(valid_idx):
                                raise ValueError("No valid non-NaN experimental dH/TdS data points to fit.")
                            # Run fit
                            model.fit_epsTS(
                                conc_T[valid_idx],
                                ddH[valid_idx],
                                TddS[valid_idx],
                                concentration_type=fit_conc_type
                            )
                            fit_progress.progress(0.5, text="Solving equilibrium with fitted epsTS...")
                            # Resolve model
                            model.solve_equil()
                            model.to_pandas()
                            fit_progress.progress(1.0, text="Fit & Simulation updated!")
                            
                            # Save state
                            st.session_state["fitted_epsTS"] = model.epsTS
                            st.session_state["fit_updated"] = True
                            st.session_state["solved_model"] = model
                            st.session_state["solved_model_type"] = model_type
                            
                            st.session_state["fit_success_msg"] = f"Successfully fitted epsTS: {model.epsTS:.4f}"
                            if hasattr(model, "resTS") and hasattr(model.resTS, "success") and not model.resTS.success:
                                msg = getattr(model.resTS, "message", "Optimizer did not converge.")
                                st.session_state["fit_warning_msg"] = f"⚠️ Optimization Warning: {msg}"
                            else:
                                st.session_state["fit_warning_msg"] = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Fitting error: {e}")
                    else:
                        st.error("Please upload experimental dH and TdS data first!")
                        
            # Display current fitted values
            st.markdown("### Current Fitted Parameters")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Fitted eps", f"{model.eps:.4f}" if st.session_state["fitted_eps"] is not None else "None")
            with col_m2:
                st.metric("Fitted epsTS", f"{model.epsTS:.4f}" if st.session_state["fitted_epsTS"] is not None else "None")
            
            # Download fitted parameters button
            has_fit = st.session_state["fitted_eps"] is not None or st.session_state["fitted_epsTS"] is not None
            if has_fit:
                fit_csv = export.get_fitted_parameters_csv(model_type, st.session_state)
                st.download_button(
                    "📥 Download Fitted Parameters (CSV)",
                    data=fit_csv,
                    file_name="fh_crowding_binary_fit_parameters.csv",
                    mime="text/csv",
                    key="download_fit_params_binary",
                    use_container_width=True
                )
        
        # Ternary fitting controls
        else:
            col_t1, col_t2 = st.columns(2)
            
            # Button 1: Fit eps
            with col_t1:
                st.markdown("**Free Energy Soft Parameters**")
                fit_eps_btn_tern = st.button("Fit eps2 & eps3 (from dG)", key="btn_fit_eps_tern", use_container_width=True)
                if fit_eps_btn_tern:
                    if (st.session_state.get("exp_val_G") is not None and 
                        st.session_state.get("exp_conc2") is not None and 
                        st.session_state.get("exp_conc3") is not None):
                        fit_progress = st.progress(0, text="Fitting eps2 & eps3...")
                        try:
                            # Filter NaNs
                            conc2 = np.array(st.session_state["exp_conc2"])
                            conc3 = np.array(st.session_state["exp_conc3"])
                            val_G = np.array(st.session_state["exp_val_G"])
                            valid_idx = np.isfinite(conc2) & np.isfinite(conc3) & np.isfinite(val_G)
                            if not np.any(valid_idx):
                                raise ValueError("No valid non-NaN experimental dG data points to fit.")
                            # Run fit
                            model.fit_eps(
                                conc2[valid_idx],
                                conc3[valid_idx],
                                val_G[valid_idx],
                                concentration_type=fit_conc_type
                            )
                            fit_progress.progress(0.5, text="Solving equilibrium with fitted eps parameters...")
                            # Resolve model
                            model.solve_equil(print_msg=False)
                            model.to_pandas()
                            fit_progress.progress(1.0, text="Fit & Simulation updated!")
                            
                            # Save state
                            st.session_state["fitted_eps2"] = model.eps2
                            st.session_state["fitted_eps3"] = model.eps3
                            st.session_state["fit_updated"] = True
                            st.session_state["solved_model"] = model
                            st.session_state["solved_model_type"] = model_type
                            
                            st.session_state["fit_success_msg"] = f"Successfully fitted eps2: {model.eps2:.4f}, eps3: {model.eps3:.4f}"
                            if hasattr(model, "res") and hasattr(model.res, "success") and not model.res.success:
                                msg = getattr(model.res, "message", "Optimizer did not converge.")
                                st.session_state["fit_warning_msg"] = f"⚠️ Optimization Warning: {msg}"
                            else:
                                st.session_state["fit_warning_msg"] = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Fitting error: {e}")
                    else:
                        st.error("Please upload experimental Ternary dG data first!")
            
            # Button 2: Fit epsTS
            with col_t2:
                st.markdown("**Entropic Soft Parameters**")
                fit_epsts_btn_tern = st.button("Fit epsTS2 & epsTS3 (from dH, TdS)", key="btn_fit_epsts_tern", use_container_width=True)
                if fit_epsts_btn_tern:
                    if (st.session_state.get("exp_val_H") is not None and 
                        st.session_state.get("exp_val_S") is not None and 
                        st.session_state.get("exp_conc2") is not None and 
                        st.session_state.get("exp_conc3") is not None):
                        fit_progress = st.progress(0, text="Fitting epsTS2 & epsTS3...")
                        try:
                            # Filter NaNs
                            conc2 = np.array(st.session_state["exp_conc2"])
                            conc3 = np.array(st.session_state["exp_conc3"])
                            val_H = np.array(st.session_state["exp_val_H"])
                            val_S = np.array(st.session_state["exp_val_S"])
                            valid_idx = np.isfinite(conc2) & np.isfinite(conc3) & np.isfinite(val_H) & np.isfinite(val_S)
                            if not np.any(valid_idx):
                                raise ValueError("No valid non-NaN experimental dH/TdS data points to fit.")
                            # Run fit
                            model.fit_epsTS(
                                conc2[valid_idx],
                                conc3[valid_idx],
                                val_H[valid_idx],
                                val_S[valid_idx],
                                concentration_type=fit_conc_type
                            )
                            fit_progress.progress(0.5, text="Solving equilibrium with fitted epsTS parameters...")
                            # Resolve model
                            model.solve_equil(print_msg=False)
                            model.to_pandas()
                            fit_progress.progress(1.0, text="Fit & Simulation updated!")
                            
                            # Save state
                            st.session_state["fitted_epsTS2"] = model.epsTS2
                            st.session_state["fitted_epsTS3"] = model.epsTS3
                            st.session_state["fit_updated"] = True
                            st.session_state["solved_model"] = model
                            st.session_state["solved_model_type"] = model_type
                            
                            st.session_state["fit_success_msg"] = f"Successfully fitted epsTS2: {model.epsTS2:.4f}, epsTS3: {model.epsTS3:.4f}"
                            if hasattr(model, "resTS") and hasattr(model.resTS, "success") and not model.resTS.success:
                                msg = getattr(model.resTS, "message", "Optimizer did not converge.")
                                st.session_state["fit_warning_msg"] = f"⚠️ Optimization Warning: {msg}"
                            else:
                                st.session_state["fit_warning_msg"] = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Fitting error: {e}")
                    else:
                        st.error("Please upload experimental Ternary dH and TdS data first!")
                        
            # Display current fitted values
            st.markdown("### Current Fitted Parameters")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Fitted eps2", f"{model.eps2:.4f}" if st.session_state["fitted_eps2"] is not None else "None")
                st.metric("Fitted eps3", f"{model.eps3:.4f}" if st.session_state["fitted_eps3"] is not None else "None")
            with col_m2:
                st.metric("Fitted epsTS2", f"{model.epsTS2:.4f}" if st.session_state["fitted_epsTS2"] is not None else "None")
                st.metric("Fitted epsTS3", f"{model.epsTS3:.4f}" if st.session_state["fitted_epsTS3"] is not None else "None")
                
            # Download fitted parameters button
            has_fit = (st.session_state["fitted_eps2"] is not None or 
                       st.session_state["fitted_eps3"] is not None or 
                       st.session_state["fitted_epsTS2"] is not None or 
                       st.session_state["fitted_epsTS3"] is not None)
            if has_fit:
                fit_csv = export.get_fitted_parameters_csv(model_type, st.session_state)
                st.download_button(
                    "📥 Download Fitted Parameters (CSV)",
                    data=fit_csv,
                    file_name="fh_crowding_ternary_fit_parameters.csv",
                    mime="text/csv",
                    key="download_fit_params_ternary",
                    use_container_width=True
                )


# ---------------------------------------------------------------------------
# Section 3: Dynamic Plotting & Visualization (Visible if solved)
# ---------------------------------------------------------------------------
if "solved_model" in st.session_state and st.session_state["solved_model_type"] == model_type:
    st.markdown("---")
    st.header("📈 Visualization & Plots")
    
    solved_model = st.session_state["solved_model"]
    
    # Checkbox to overlay experimental data (shown only if some exp data is uploaded)
    show_exp = False
    if st.session_state["exp_data_loaded"]:
        show_exp = st.checkbox("Overlay uploaded experimental data on plots", value=True, key="show_exp_data")
        
    plot_option = st.selectbox(
        "Select Plotting Mode",
        ["Standard Preset Plot", "Custom Axis Plot"]
    )
    
    if plot_option == "Standard Preset Plot":
        if model_type == "Binary Crowding Model":
            st.subheader("Standard 3x3 Results Plot")
            col1, col2 = st.columns(2)
            with col1:
                conc_type_plot = st.selectbox("Concentration axis type", ["phi", "molar", "molal"], key="bin_plot_conc")
            with col2:
                plot_unit = st.selectbox("Plotting Unit", ["kJ/mol", "kcal/mol"], key="bin_plot_unit")
            
            # Setup experimental values to overlay if enabled
            folding_plot = (plot_unit == "kJ/mol")
            plot_kwargs = {"concentration_type": conc_type_plot, "folding": folding_plot}
            if show_exp:
                # Convert concentrations dynamically to match plotted unit
                if st.session_state.get("exp_conc_G") is not None:
                    plot_kwargs["exp_conc"] = convert_exp_conc(
                        st.session_state["exp_conc_G"],
                        from_type=uploaded_conc_unit,
                        to_type=conc_type_plot if conc_type_plot != "phi" else "phiC",
                        model=solved_model
                    )
                    plot_kwargs["exp_ddG"] = st.session_state.get("exp_ddG", np.nan)
                    plot_kwargs["err_ddG"] = st.session_state.get("err_ddG", np.nan)
                    
                if st.session_state.get("exp_conc_T") is not None:
                    plot_kwargs["exp_concT"] = convert_exp_conc(
                        st.session_state["exp_conc_T"],
                        from_type=uploaded_conc_unit,
                        to_type=conc_type_plot if conc_type_plot != "phi" else "phiC",
                        model=solved_model
                    )
                    plot_kwargs["exp_ddH"] = st.session_state.get("exp_ddH", np.nan)
                    plot_kwargs["exp_TddS"] = st.session_state.get("exp_TddS", np.nan)
                    plot_kwargs["err_ddH"] = st.session_state.get("err_ddH", np.nan)
                    plot_kwargs["err_TddS"] = st.session_state.get("err_TddS", np.nan)

            try:
                plotter = fh_crowding.BinaryPlotter(solved_model)
                fig = plotter.plot_results(**plot_kwargs)
                _display_and_export_plot(fig, "fh_crowding_binary_preset_plot", "bin_preset_plot")
            except Exception as e:
                st.error(f"Error rendering preset plots: {e}")
            
        else: # Ternary
            st.subheader("Ternary Standard Presets")
            preset_plot = st.selectbox(
                "Select Standard Plot",
                [
                    "ddG (3x3 contour)",
                    "phiS (Contours of subdomain concentrations)",
                    "Ms (Contours of subdomain volume fractions)",
                    "mus2 (Contours of subdomain 2 chemical potentials)",
                    "mus3 (Contours of subdomain 3 chemical potentials)",
                    "TdS_mix (Contours of mixing entropy)",
                    "dG_mix (Contours of mixing free energy)",
                    "ddG_mu (Contours of ddG chemical potentials)",
                    "TddS (Contours of TddS entropy)",
                    "ddH (Contours of ddH enthalpy)",
                    "Gamma (Contours of preferential interaction coefficients)",
                    "Gamma_mu (Contours of preferential interaction mu)",
                    "Gamma_mu_der (Contours of preferential interaction derivatives)"
                ]
            )
            plotter = fh_crowding.TernaryPlotter(solved_model)
            
            if "ddG (3x3 contour)" in preset_plot:
                fig = plotter.plot_ddG()
            elif "phiS" in preset_plot:
                fig = plotter.plot_phiS()
            elif "Ms" in preset_plot:
                fig = plotter.plot_Ms()
            elif "mus2" in preset_plot:
                fig = plotter.plot_mus2()
            elif "mus3" in preset_plot:
                fig = plotter.plot_mus3()
            elif "TdS_mix" in preset_plot:
                fig = plotter.plot_TdS_mix()
            elif "dG_mix" in preset_plot:
                fig = plotter.plot_dG_mix()
            elif "ddG_mu" in preset_plot:
                fig = plotter.plot_ddG_mu()
            elif "TddS" in preset_plot:
                fig = plotter.plot_TddS()
            elif "ddH" in preset_plot:
                fig = plotter.plot_ddH()
            elif "Gamma" in preset_plot:
                fig = plotter.plot_Gamma()
            elif "Gamma_mu_der" in preset_plot:
                fig = plotter.plot_Gamma_mu_der()
            elif "Gamma_mu" in preset_plot:
                fig = plotter.plot_Gamma_mu()
                
            # If ternary contour preset and show_exp is enabled, overlay exp points on the subplots
            is_fold_preset = any(p in preset_plot for p in ["ddG (3x3 contour)", "TddS", "ddH"])
            if show_exp and is_fold_preset and st.session_state.get("exp_conc2") is not None:
                # Convert exp points concentration to phi
                exp_x_phi = convert_exp_conc(st.session_state["exp_conc2"], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=2)
                exp_y_phi = convert_exp_conc(st.session_state["exp_conc3"], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=3, exp_conc3=st.session_state["exp_conc3"])
                
                # Iterate axes and scatter the experimental points
                for ax in fig.get_axes():
                    # only scatter on contour subplots (skip colorbar axes which have no set_xlabel or set_ylabel)
                    if hasattr(ax, 'get_xlabel') and ax.get_xlabel() == r'$\phi_2$':
                        ax.scatter(exp_x_phi, exp_y_phi, color='red', edgecolor='white', s=25, label='Experimental', zorder=10)
                        
            try:
                _display_and_export_plot(fig, "fh_crowding_ternary_preset_plot", "tern_preset_plot")
            except Exception as e:
                st.error(f"Error rendering preset plot: {e}")

    else: # Custom Axis Plot
        if model_type == "Binary Crowding Model":
            st.subheader("Custom 1D Plot")
            
            properties = {
                "Concentration (phiC)": ("phiC", r"$\phi_C$"),
                "Molar Concentration": ("molar", "Molar [M]"),
                "Molal Concentration": ("molal", "Molal [mol/kg]"),
                "Osmotic Pressure": ("osm", r"$\Pi$ (Osmolal)"),
                "Preferential Hydration (gamma)": ("gamma", r"$\Delta\Gamma_S$"),
                "Preferential Interaction (gammaC)": ("gammaC", r"$\Delta\Gamma_C$"),
                "phiCsurf": ("phiCsurf", r"$\phi_C^{surf}$"),
                "phiSsurf": ("phiSsurf", r"$\phi_S^{surf}$"),
                "Free Energy (ddA) [kJ]": ("ddA_kj", r"$\Delta\Delta G^0$ [kJ/mol]"),
                "Free Energy (ddA) [kcal]": ("ddA_kcal", r"$\Delta\Delta G^0$ [kcal/mol]"),
                "Enthalpy (ddE) [kJ]": ("ddE_kj", r"$\Delta\Delta H^0$ [kJ/mol]"),
                "Enthalpy (ddE) [kcal]": ("ddE_kcal", r"$\Delta\Delta H^0$ [kcal/mol]"),
                "Entropy (TddS) [kJ]": ("TddS_kj", r"$T\Delta\Delta S^0$ [kJ/mol]"),
                "Entropy (TddS) [kcal]": ("TddS_kcal", r"$T\Delta\Delta S^0$ [kcal/mol]"),
            }
            
            col1, col2 = st.columns(2)
            with col1:
                x_name = st.selectbox("X-Axis Property", list(properties.keys()), index=0)
            with col2:
                y_name = st.selectbox("Y-Axis Property", list(properties.keys()), index=8)
                
            x_attr, x_label = properties[x_name]
            y_attr, y_label = properties[y_name]
            
            # Detect potential base types for unit alignment
            x_base = None
            y_base = None
            for prefix in ["ddA", "ddE", "TddS"]:
                if prefix in x_attr:
                    x_base = prefix
                if prefix in y_attr:
                    y_base = prefix
                    
            # Check if Y-Axis is one of the potentials for contribution plotting
            is_potential = False
            pot_type = None
            pot_unit = "kJ"
            if y_base is not None:
                is_potential = True
                pot_type = y_base
                pot_unit = "kcal" if "kcal" in y_attr else "kJ"
                
            # Align unit of X-axis to match Y-axis if both are thermodynamic potentials
            if x_base is not None and y_base is not None:
                if pot_unit == "kcal" and "kcal" not in x_attr:
                    x_attr = f"{x_base}_kcal"
                    # Find correct label from properties dict
                    for k, v in properties.items():
                        if v[0] == x_attr:
                            x_label = v[1]
                            break
                elif pot_unit == "kJ" and "kj" not in x_attr:
                    x_attr = f"{x_base}_kj"
                    # Find correct label from properties dict
                    for k, v in properties.items():
                        if v[0] == x_attr:
                            x_label = v[1]
                            break

            plot_contrib = False
            if is_potential and x_base != y_base:
                plot_contrib = st.checkbox("Plot alongside contributions (nu, chi, eps)", value=True)
                
            fig, ax = plt.subplots(figsize=(6, 4))
            x_data = getattr(solved_model, x_attr)
            
            if plot_contrib:
                if pot_type == "ddA":
                    if pot_unit == "kJ":
                        ax.plot(x_data, solved_model.ddA_kj, label="Total Model", color="black", linewidth=2)
                        ax.plot(x_data, solved_model.ddA_nu_kj, label=r"$\nu$ (Excluded Volume)", linestyle="--")
                        ax.plot(x_data, solved_model.ddA_chi_kj, label=r"$\chi$ (Non-ideal mixing)", linestyle=":")
                        ax.plot(x_data, solved_model.ddA_eps_kj, label=r"$\varepsilon$ (Soft interaction)", linestyle="-.")
                    else:
                        ax.plot(x_data, solved_model.ddA_kcal, label="Total Model", color="black", linewidth=2)
                        ax.plot(x_data, solved_model.ddA_nu_kcal, label=r"$\nu$ (Excluded Volume)", linestyle="--")
                        ax.plot(x_data, solved_model.ddA_chi_kcal, label=r"$\chi$ (Non-ideal mixing)", linestyle=":")
                        ax.plot(x_data, solved_model.ddA_eps_kcal, label=r"$\varepsilon$ (Soft interaction)", linestyle="-.")
                elif pot_type == "ddE":
                    if pot_unit == "kJ":
                        ax.plot(x_data, solved_model.ddE_kj, label="Total Model", color="black", linewidth=2)
                        ax.plot(x_data, solved_model.ddE_chi_kj, label=r"$\chi$ (Non-ideal mixing)", linestyle=":")
                        ax.plot(x_data, solved_model.ddE_eps_kj, label=r"$\varepsilon$ (Soft interaction)", linestyle="-.")
                    else:
                        ax.plot(x_data, solved_model.ddE_kcal, label="Total Model", color="black", linewidth=2)
                        ax.plot(x_data, solved_model.ddE_chi_kcal, label=r"$\chi$ (Non-ideal mixing)", linestyle=":")
                        ax.plot(x_data, solved_model.ddE_eps_kcal, label=r"$\varepsilon$ (Soft interaction)", linestyle="-.")
                elif pot_type == "TddS":
                    if pot_unit == "kJ":
                        ax.plot(x_data, solved_model.TddS_kj, label="Total Model", color="black", linewidth=2)
                        ax.plot(x_data, solved_model.TddS_nu_kj, label=r"$\nu$ (Excluded Volume)", linestyle="--")
                        ax.plot(x_data, solved_model.TddS_chi_kj, label=r"$\chi$ (Non-ideal mixing)", linestyle=":")
                        ax.plot(x_data, solved_model.TddS_eps_kj, label=r"$\varepsilon$ (Soft interaction)", linestyle="-.")
                    else:
                        ax.plot(x_data, solved_model.TddS_kcal, label="Total Model", color="black", linewidth=2)
                        ax.plot(x_data, solved_model.TddS_nu_kcal, label=r"$\nu$ (Excluded Volume)", linestyle="--")
                        ax.plot(x_data, solved_model.TddS_chi_kcal, label=r"$\chi$ (Non-ideal mixing)", linestyle=":")
                        ax.plot(x_data, solved_model.TddS_eps_kcal, label=r"$\varepsilon$ (Soft interaction)", linestyle="-.")
                ax.legend()
            else:
                y_data = getattr(solved_model, y_attr)
                ax.plot(x_data, y_data, color="black", linewidth=2, label="Model results")
            # Overlay custom experimental points if requested and applicable
            if show_exp:
                exp_x = None
                exp_y = None
                err_y = None
                y_conc = None
                
                # Identify dataset and Y-values based on Y-axis attribute
                if "ddA" in y_attr:
                    exp_y = st.session_state.get("exp_ddG")
                    err_y = st.session_state.get("err_ddG")
                    y_conc = st.session_state.get("exp_conc_G")
                    if "kcal" in y_attr and exp_y is not None:
                        exp_y = exp_y / 4.184
                        if err_y is not None:
                            err_y = err_y / 4.184
                elif "ddE" in y_attr:
                    exp_y = st.session_state.get("exp_ddH")
                    err_y = st.session_state.get("err_ddH")
                    y_conc = st.session_state.get("exp_conc_T")
                    if "kcal" in y_attr and exp_y is not None:
                        exp_y = exp_y / 4.184
                        if err_y is not None:
                            err_y = err_y / 4.184
                elif "TddS" in y_attr:
                    exp_y = st.session_state.get("exp_TddS")
                    err_y = st.session_state.get("err_TddS")
                    y_conc = st.session_state.get("exp_conc_T")
                    if "kcal" in y_attr and exp_y is not None:
                        exp_y = exp_y / 4.184
                        if err_y is not None:
                            err_y = err_y / 4.184
                            
                # Determine experimental X-values based on X-axis attribute and Y-axis's concentration points
                if exp_y is not None and y_conc is not None:
                    if x_attr in ["phiC", "molar", "molal"]:
                        exp_x = convert_exp_conc(
                            y_conc,
                            from_type=uploaded_conc_unit,
                            to_type=x_attr if x_attr != "phi" else "phiC",
                            model=solved_model
                        )
                    elif "ddA" in x_attr:
                        ref_g = st.session_state.get("exp_conc_G")
                        val_g = st.session_state.get("exp_ddG")
                        if ref_g is not None and val_g is not None:
                            sort_idx = np.argsort(ref_g)
                            exp_x = np.interp(y_conc, ref_g[sort_idx], val_g[sort_idx])
                            if "kcal" in x_attr:
                                exp_x = exp_x / 4.184
                    elif "ddE" in x_attr:
                        ref_t = st.session_state.get("exp_conc_T")
                        val_h = st.session_state.get("exp_ddH")
                        if ref_t is not None and val_h is not None:
                            sort_idx = np.argsort(ref_t)
                            exp_x = np.interp(y_conc, ref_t[sort_idx], val_h[sort_idx])
                            if "kcal" in x_attr:
                                exp_x = exp_x / 4.184
                    elif "TddS" in x_attr:
                        ref_t = st.session_state.get("exp_conc_T")
                        val_s = st.session_state.get("exp_TddS")
                        if ref_t is not None and val_s is not None:
                            sort_idx = np.argsort(ref_t)
                            exp_x = np.interp(y_conc, ref_t[sort_idx], val_s[sort_idx])
                            if "kcal" in x_attr:
                                exp_x = exp_x / 4.184
                                
                if exp_y is not None and exp_x is not None:
                    # Scatter experimental data points
                    if err_y is not None and not np.all(np.isnan(err_y)):
                        ax.errorbar(
                            exp_x, exp_y, yerr=err_y,
                            fmt='o', color='red', ecolor='red', capsize=4,
                            label='Experimental', zorder=5
                        )
                    else:
                        ax.scatter(
                            exp_x, exp_y,
                            color='red', edgecolor='black', s=45,
                            label='Experimental', zorder=5
                        )
                    ax.legend()
                    
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, linestyle=":", alpha=0.6)
            try:
                _display_and_export_plot(fig, "fh_crowding_binary_custom_plot", "bin_custom_plot")
            except Exception as e:
                st.error(f"Error rendering custom 1D plot: {e}")
            
        else: # Ternary Custom Plot
            st.subheader("Custom Ternary Plotting")
            
            tern_mode = st.radio("Plot Type", ["2D Contour Plot", "1D Slice Plot"])
            
            properties_contour = {
                "Free Energy (ddG) [kT]": ("ddG", r"$\Delta\Delta G^0 / (k_B T)$"),
                "Free Energy (ddG) [kJ]": ("ddG_kJ", r"$\Delta\Delta G^0$ [kJ/mol]"),
                "Enthalpy (ddH) [kT]": ("ddH", r"$\Delta\Delta H^0 / (k_B T)$"),
                "Enthalpy (ddH) [kJ]": ("ddH_kJ", r"$\Delta\Delta H^0$ [kJ/mol]"),
                "Entropy (TddS) [kT]": ("TddS", r"$T\Delta\Delta S^0 / (k_B T)$"),
                "Entropy (TddS) [kJ]": ("TddS_kJ", r"$T\Delta\Delta S^0$ [kJ/mol]"),
                "Osmotic Pressure": ("osm", r"$\Pi$ (Osmolal)"),
                "Preferential Interaction 2 (Gamma_2)": ("Gamma_2", r"$\Delta\Gamma_2$"),
                "Preferential Interaction 3 (Gamma_3)": ("Gamma_3", r"$\Delta\Gamma_3$"),
                "Preferential Interaction 1,2 (Gamma_1_2)": ("Gamma_1_2", r"$\Delta\Gamma_{1,2}$"),
                "Preferential Interaction 1,3 (Gamma_1_3)": ("Gamma_1_3", r"$\Delta\Gamma_{1,3}$"),
            }
            
            if tern_mode == "2D Contour Plot":
                z_name = st.selectbox("Property to plot (Contours over phi2 vs phi3)", list(properties_contour.keys()))
                z_attr, z_label = properties_contour[z_name]
                
                is_potential = any(p in z_attr for p in ["ddG", "ddH", "TddS"])
                plot_contrib = False
                if is_potential:
                    plot_contrib = st.checkbox("Plot alongside contributions (nu, chi, eps)", value=True)
                    
                if plot_contrib:
                    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(9, 7), layout="constrained")
                    
                    if "ddG" in z_attr:
                        base = "ddG"
                    elif "ddH" in z_attr:
                        base = "ddH"
                    else:
                        base = "TddS"
                        
                    suffix = "_kJ" if "_kJ" in z_attr else ""
                    
                    # Subplot 1: Total
                    total_z = getattr(solved_model, f"{base}{suffix}")
                    cp = axes[0,0].contourf(solved_model.phi2, solved_model.phi3, total_z, levels=12)
                    axes[0,0].contour(cp, colors='k', linestyles='solid', linewidths=0.5)
                    axes[0,0].set_title(f"Total {z_label}")
                    fig.colorbar(cp, ax=axes[0,0])
                    
                    # Subplot 2: nu
                    if base == "ddH":
                        axes[0,1].text(0.5, 0.5, "No Excluded Volume (\n$\\nu$) contribution\nfor Enthalpy", 
                                       ha='center', va='center', fontsize=12)
                        axes[0,1].set_title(r"$\nu$ Contribution")
                    else:
                        nu_z = getattr(solved_model, f"{base}_nu{suffix}")
                        cp = axes[0,1].contourf(solved_model.phi2, solved_model.phi3, nu_z, levels=12)
                        axes[0,1].contour(cp, colors='k', linestyles='solid', linewidths=0.5)
                        axes[0,1].set_title(rf"$\nu$ Contribution")
                        fig.colorbar(cp, ax=axes[0,1])
                        
                    # Subplot 3: chi
                    chi_z = getattr(solved_model, f"{base}_chi{suffix}")
                    cp = axes[1,0].contourf(solved_model.phi2, solved_model.phi3, chi_z, levels=12)
                    axes[1,0].contour(cp, colors='k', linestyles='solid', linewidths=0.5)
                    axes[1,0].set_title(rf"$\chi$ Contribution")
                    fig.colorbar(cp, ax=axes[1,0])
                    
                    # Subplot 4: eps
                    eps_z = getattr(solved_model, f"{base}_eps{suffix}")
                    cp = axes[1,1].contourf(solved_model.phi2, solved_model.phi3, eps_z, levels=12)
                    axes[1,1].contour(cp, colors='k', linestyles='solid', linewidths=0.5)
                    axes[1,1].set_title(rf"$\varepsilon$ Contribution")
                    fig.colorbar(cp, ax=axes[1,1])
                    
                    # Overlay Ternary Experimental Points on Subplots if requested
                    if show_exp and st.session_state.get("exp_conc2") is not None:
                        exp_x_phi = convert_exp_conc(st.session_state["exp_conc2"], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=2)
                        exp_y_phi = convert_exp_conc(st.session_state["exp_conc3"], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=3, exp_conc3=st.session_state["exp_conc3"])
                        for row in axes:
                            for ax in row:
                                # overlay only on contour axes
                                if ax.get_xlabel() == "":
                                    ax.scatter(exp_x_phi, exp_y_phi, color='red', edgecolor='white', s=25, label='Experimental', zorder=10)
                    
                    for ax_row in axes:
                        for ax in ax_row:
                            ax.set_xlabel(r"$\phi_2$")
                            ax.set_ylabel(r"$\phi_3$")
                    try:
                        _display_and_export_plot(fig, "fh_crowding_ternary_contour_subplots", "tern_contour_subplots")
                    except Exception as e:
                        st.error(f"Error rendering custom 2D contour plot subplots: {e}")
                else:
                    fig, ax = plt.subplots(figsize=(6, 4.5))
                    z_data = getattr(solved_model, z_attr)
                    cp = ax.contourf(solved_model.phi2, solved_model.phi3, z_data, levels=15)
                    ax.contour(cp, colors='k', linestyles='solid', linewidths=0.5)
                    ax.set_xlabel(r"$\phi_2$")
                    ax.set_ylabel(r"$\phi_3$")
                    ax.set_title(z_label)
                    fig.colorbar(cp)
                    
                    # Overlay experimental points on single contour
                    if show_exp and is_potential and st.session_state.get("exp_conc2") is not None:
                        exp_x_phi = convert_exp_conc(st.session_state["exp_conc2"], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=2)
                        exp_y_phi = convert_exp_conc(st.session_state["exp_conc3"], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=3, exp_conc3=st.session_state["exp_conc3"])
                        ax.scatter(exp_x_phi, exp_y_phi, color='red', edgecolor='white', s=45, label='Experimental', zorder=10)
                        ax.legend()
                        
                    try:
                        _display_and_export_plot(fig, "fh_crowding_ternary_contour_plot", "tern_contour_plot")
                    except Exception as e:
                        st.error(f"Error rendering custom 2D contour plot: {e}")
                    
            else: # 1D Slice Plot
                st.write("Slices through the 2D concentration space to make 1D plots.")
                
                phi2_axis = solved_model.phi2[0, :]
                phi3_axis = solved_model.phi3[:, 0]
                
                slice_type = st.selectbox(
                    "Select Slice Path",
                    ["Constant phi3", "Constant phi2", "Diagonal (phi2 = phi3)"]
                )
                
                if slice_type == "Constant phi3":
                    val3 = st.select_slider("Select constant phi3 value", options=list(phi3_axis))
                    idx3 = np.where(phi3_axis == val3)[0][0]
                    x_data = phi2_axis
                    x_label = r"$\phi_2$"
                    slicer = lambda arr2d: arr2d[idx3, :]
                elif slice_type == "Constant phi2":
                    val2 = st.select_slider("Select constant phi2 value", options=list(phi2_axis))
                    idx2 = np.where(phi2_axis == val2)[0][0]
                    x_data = phi3_axis
                    x_label = r"$\phi_3$"
                    slicer = lambda arr2d: arr2d[:, idx2]
                else: # Diagonal
                    x_data = np.diag(solved_model.phi2)
                    x_label = r"$\phi_2 = \phi_3$"
                    slicer = lambda arr2d: np.diag(arr2d)
                    
                y_name = st.selectbox("Y-Axis Property", list(properties_contour.keys()))
                y_attr, y_label = properties_contour[y_name]
                
                is_potential = any(p in y_attr for p in ["ddG", "ddH", "TddS"])
                plot_contrib = False
                if is_potential:
                    plot_contrib = st.checkbox("Plot alongside contributions (nu, chi, eps)", value=True, key="tern_slice_contrib")
                    
                fig, ax = plt.subplots(figsize=(6, 4))
                
                if plot_contrib:
                    if "ddG" in y_attr:
                        base = "ddG"
                    elif "ddH" in y_attr:
                        base = "ddH"
                    else:
                        base = "TddS"
                    suffix = "_kJ" if "_kJ" in y_attr else ""
                    
                    total_z = getattr(solved_model, f"{base}{suffix}")
                    ax.plot(x_data, slicer(total_z), label="Total Model", color="black", linewidth=2)
                    
                    if base != "ddH":
                        nu_z = getattr(solved_model, f"{base}_nu{suffix}")
                        ax.plot(x_data, slicer(nu_z), label=r"$\nu$ (Excluded Volume)", linestyle="--")
                        
                    chi_z = getattr(solved_model, f"{base}_chi{suffix}")
                    ax.plot(x_data, slicer(chi_z), label=r"$\chi$ (Non-ideal mixing)", linestyle=":")
                    
                    eps_z = getattr(solved_model, f"{base}_eps{suffix}")
                    ax.plot(x_data, slicer(eps_z), label=r"$\varepsilon$ (Soft interaction)", linestyle="-.")
                    
                    ax.legend()
                else:
                    y_data = getattr(solved_model, y_attr)
                    ax.plot(x_data, slicer(y_data), color="black", linewidth=2, label="Model results")
                    
                # Overlay experimental data on 1D Slice if requested
                if show_exp and st.session_state.get("exp_conc2") is not None:
                    # We need the appropriate Y array to slice
                    if "ddG" in y_attr:
                        exp_val = st.session_state.get("exp_val_G")
                    elif "ddH" in y_attr:
                        exp_val = st.session_state.get("exp_val_H")
                    elif "TddS" in y_attr:
                        exp_val = st.session_state.get("exp_val_S")
                    else:
                        exp_val = None
                        
                    if exp_val is not None:
                        if "kcal" in y_attr:
                            exp_val = exp_val / 4.184
                        # Convert exp concentrations to volume fractions
                        exp_x_phi = convert_exp_conc(st.session_state["exp_conc2"], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=2)
                        exp_y_phi = convert_exp_conc(st.session_state["exp_conc3"], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=3, exp_conc3=st.session_state["exp_conc3"])
                        
                        # Find points in experimental dataset that align with the slice path (within tolerance)
                        tolerance = 0.005
                        if slice_type == "Constant phi3":
                            mask = np.abs(exp_y_phi - val3) < tolerance
                            slice_exp_x = exp_x_phi[mask]
                            slice_exp_y = exp_val[mask]
                        elif slice_type == "Constant phi2":
                            mask = np.abs(exp_x_phi - val2) < tolerance
                            slice_exp_x = exp_y_phi[mask]  # plot against phi3 axis
                            slice_exp_y = exp_val[mask]
                        else: # Diagonal
                            mask = np.abs(exp_x_phi - exp_y_phi) < tolerance
                            slice_exp_x = exp_x_phi[mask]
                            slice_exp_y = exp_val[mask]
                            
                        if len(slice_exp_x) > 0:
                            ax.scatter(slice_exp_x, slice_exp_y, color='red', edgecolor='black', s=45, label='Experimental', zorder=5)
                            ax.legend()
                            
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.grid(True, linestyle=":", alpha=0.6)
                try:
                    _display_and_export_plot(fig, "fh_crowding_ternary_slice_plot", "tern_slice_plot")
                except Exception as e:
                    st.error(f"Error rendering custom 1D slice plot: {e}")
