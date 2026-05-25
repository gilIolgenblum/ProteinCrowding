import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import sys
import base64
from pathlib import Path

# Ensure local directories are in path for Streamlit Cloud
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))

import fh_crowding
import session_io
import export
import validation
import styles
import citation

st.set_page_config(
    page_title="FH Crowding Model — Thermodynamic Analysis",
    layout="wide",
    menu_items={"About": "FH Crowding Thermodynamic Model. Built with Streamlit."}
)
styles.inject_css()

# ---------------------------------------------------------------------------
# Global matplotlib style for all preset plots
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":       9,
    "axes.linewidth":  0.8,
    "axes.labelsize":  9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
    "figure.dpi":     120,
})

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


def _display_and_export_plotly(fig, filename, key, height=500):
    """Render a Plotly figure with MathJax support and provide PNG/SVG download buttons."""
    html_str = pio.to_html(fig, include_mathjax='cdn', full_html=False)
    components.html(html_str, height=height, scrolling=False)
    col1, col2 = st.columns(2)
    with col1:
        try:
            png_bytes = export.plotly_fig_to_bytes(fig, "png")
            st.download_button(
                "📥 Download Plot (PNG)",
                data=png_bytes,
                file_name=f"{filename}.png",
                mime="image/png",
                key=f"{key}_png",
                use_container_width=True
            )
        except Exception:
            st.caption("PNG export requires kaleido (`pip install kaleido`).")
    with col2:
        try:
            svg_bytes = export.plotly_fig_to_bytes(fig, "svg")
            st.download_button(
                "📥 Download Plot (SVG)",
                data=svg_bytes,
                file_name=f"{filename}.svg",
                mime="image/svg+xml",
                key=f"{key}_svg",
                use_container_width=True
            )
        except Exception:
            st.caption("SVG export unavailable.")

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
    file_path = Path(__file__).parent / "sample_data" / sample_files[sample_sel]
    try:
        df = pd.read_csv(file_path)
        df["concentration"] = pd.to_numeric(df["concentration"], errors='coerce')
        df = df.dropna(subset=["concentration"])
        df.loc[df["concentration"] <= 0.0, "concentration"] = 0.0001
        
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
        st.session_state["exp_conc2_G"] = None
        st.session_state["exp_conc3_G"] = None
        st.session_state["exp_conc2_T"] = None
        st.session_state["exp_conc3_T"] = None
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
    files = sample_files[sample_sel]
    try:
        app_dir = Path(__file__).parent
        df_G = pd.read_csv(app_dir / "sample_data" / files["dG"])
        df_G["conc2"] = pd.to_numeric(df_G["conc2"], errors='coerce')
        df_G["conc3"] = pd.to_numeric(df_G["conc3"], errors='coerce')
        df_G["dG"] = pd.to_numeric(df_G["dG"], errors='coerce')
        df_G = df_G.dropna(subset=["conc2", "conc3", "dG"])
        df_G.loc[df_G["conc2"] <= 0.0, "conc2"] = 0.0001
        df_G.loc[df_G["conc3"] <= 0.0, "conc3"] = 0.0001
        
        st.session_state["exp_conc2"] = df_G["conc2"].values
        st.session_state["exp_conc3"] = df_G["conc3"].values
        st.session_state["exp_conc2_G"] = df_G["conc2"].values
        st.session_state["exp_conc3_G"] = df_G["conc3"].values
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
            df_H = pd.read_csv(app_dir / "sample_data" / files["dH"])
            df_H["conc2"] = pd.to_numeric(df_H["conc2"], errors='coerce')
            df_H["conc3"] = pd.to_numeric(df_H["conc3"], errors='coerce')
            df_H["dH"] = pd.to_numeric(df_H["dH"], errors='coerce')
            df_H = df_H.dropna(subset=["conc2", "conc3", "dH"])
            df_H.loc[df_H["conc2"] <= 0.0, "conc2"] = 0.0001
            df_H.loc[df_H["conc3"] <= 0.0, "conc3"] = 0.0001
            st.session_state["raw_exp_val_H"] = df_H["dH"].values
            st.session_state["exp_val_H"] = df_H["dH"].values
            st.session_state["exp_conc2_T"] = df_H["conc2"].values
            st.session_state["exp_conc3_T"] = df_H["conc3"].values
        except Exception:
            pass
            
        try:
            df_S = pd.read_csv(app_dir / "sample_data" / files["TdS"])
            df_S["conc2"] = pd.to_numeric(df_S["conc2"], errors='coerce')
            df_S["conc3"] = pd.to_numeric(df_S["conc3"], errors='coerce')
            df_S["TdS"] = pd.to_numeric(df_S["TdS"], errors='coerce')
            df_S = df_S.dropna(subset=["conc2", "conc3", "TdS"])
            df_S.loc[df_S["conc2"] <= 0.0, "conc2"] = 0.0001
            df_S.loc[df_S["conc3"] <= 0.0, "conc3"] = 0.0001
            st.session_state["raw_exp_val_S"] = df_S["TdS"].values
            st.session_state["exp_val_S"] = df_S["TdS"].values
            if st.session_state.get("exp_conc2_T") is None:
                st.session_state["exp_conc2_T"] = df_S["conc2"].values
                st.session_state["exp_conc3_T"] = df_S["conc3"].values
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
    "tern_dphi2": 0.001, "tern_phi2_max": 0.2,
    
    # ternary cosolute 3
    "nu3":       1.0,  "chi13":     0.1,  "chiTS13":    -0.05,
    "tern_dphi3": 0.001, "tern_phi3_max": 0.2,
    
    # ternary cross-interaction
    "chi23":     0.0,  "chiTS23":   0.0,
    "eps23":     0.0,  "epsTS23":   0.0,
    
    # experimental data keys
    "exp_conc_G": None, "exp_ddG": None, "err_ddG": None,
    "exp_conc_T": None, "exp_ddH": None, "exp_TddS": None, "err_ddH": None, "err_TddS": None,
    "exp_conc2": None, "exp_conc3": None,
    "exp_conc2_G": None, "exp_conc3_G": None,
    "exp_conc2_T": None, "exp_conc3_T": None,
    "exp_val_G": None, "exp_val_H": None, "exp_val_S": None,
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
    "fitted_eps2": None, "fitted_eps3": None, "fitted_eps23": None,
    "fitted_epsTS2": None, "fitted_epsTS3": None, "fitted_epsTS23": None,
    "fit_updated": False,
    "fit_success_msg": None,
    "fit_warning_msg": None,
    "is_fitting_mode": False
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
    if st.session_state.get("fitted_eps23") is not None:
        st.session_state["eps23"] = st.session_state["fitted_eps23"]
    if st.session_state.get("fitted_epsTS2") is not None:
        st.session_state["tern_epsts2_input"] = st.session_state["fitted_epsTS2"]
    if st.session_state.get("fitted_epsTS3") is not None:
        st.session_state["tern_epsts3_input"] = st.session_state["fitted_epsTS3"]
    if st.session_state.get("fitted_epsTS23") is not None:
        st.session_state["epsTS23"] = st.session_state["fitted_epsTS23"]
    st.session_state["fit_updated"] = False


# ---------------------------------------------------------------------------
# Sidebar — common
# ---------------------------------------------------------------------------
LOGO_PATH = Path(__file__).parent / "logo.png"
logo_b64 = None
if LOGO_PATH.exists():
    with open(LOGO_PATH, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode("utf-8")

logo_col, title_col = st.columns([1, 6])

with logo_col:
    if logo_b64:
        # Added negative margin to vertically align with the title nicely
        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown(f'<img src="data:image/png;base64,{logo_b64}" width="100" style="object-fit: contain;">', unsafe_allow_html=True)

with title_col:
    st.markdown(
        """
        <div style="margin-bottom:0.3rem;">
          <h1 style="font-size:1.7rem; font-weight:700; color:#2C3E50; margin-bottom:0.1rem; margin-top:0;">
            FH Crowding Thermodynamic Model
          </h1>
          <p style="font-size:0.87rem; color:#5D7A8A; margin-top:0;">
            Flory–Huggins model for protein–cosolute crowding thermodynamics.
            Choose a mode below to simulate or fit experimental data.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
styles.workflow_banner()

st.sidebar.markdown(
    """
    <div style="margin-top:1.0rem;">
      <h2 style="font-size:1.1rem; font-weight:600; color:#2C3E50;">Model Parameters</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<p style="font-size:0.78rem; color:#5D7A8A; font-weight:600; '
    'text-transform:uppercase; letter-spacing:0.05em;">Model Configuration</p>',
    unsafe_allow_html=True,
)
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
    st.session_state["fitted_eps23"] = None
    st.session_state["fitted_epsTS2"] = None
    st.session_state["fitted_epsTS3"] = None
    st.session_state["fitted_epsTS23"] = None

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

    nu    = st.sidebar.number_input("ν (excluded volume)",              key="bin_nu",    step=0.01, format="%.4f")
    chi   = st.sidebar.number_input("χ (non-ideal mixing)",              key="bin_chi",   step=0.01, format="%.4f")
    chiTS = st.sidebar.number_input("χₜₛ (entropy component of χ)",      key="bin_chiTS", step=0.01, format="%.4f")

    st.sidebar.subheader("Soft Interaction (ε)")
    
    eps   = st.sidebar.number_input("ε (soft interaction)",              step=0.01, format="%.4f", key="bin_eps_input")
    epsTS = st.sidebar.number_input("εTS (entropy component of ε)",     step=0.01, format="%.4f", key="bin_epsts_input")

    st.sidebar.subheader("Simulation Grid")
    
    bin_advanced = st.sidebar.checkbox(
        "Enable Advanced/High-Resolution Grid",
        value=False,
        key="bin_advanced_grid",
        help="Allows extremely fine grids. Warning: may crash the app if memory limits are exceeded."
    )
    
    bin_min_dphi = 1e-5 if bin_advanced else 0.001
    
    dphiC = st.sidebar.number_input(
        "Δϕᶜ (grid step)",
        min_value=bin_min_dphi,
        max_value=0.05,
        step=0.0005,
        format="%.5f",
        key="bin_dphiC",
        help="Concentration grid step size. Smaller = finer grid, slower simulation. "
             "Package default: 0.0001. App default: 0.001 (10× faster).",
    )
    phiC_max = st.sidebar.number_input(
        "ϕᶜ max",
        min_value=0.001,
        max_value=1.0,
        step=0.01,
        format="%.3f",
        key="bin_phiC_max",
        help="Maximum volume fraction of the grid. Default: 0.15.",
    )

    n_points = validation.estimate_binary_grid_points(0.0001, phiC_max, dphiC)
    n_points, mem_mb = validation.validate_grid_size(n_points, bin_advanced, num_arrays=5)
    st.sidebar.caption(f"Grid: {n_points:,} pts | Est. Mem: ~{mem_mb:.1f} MB")

    try:
        cosolute = fh_crowding.Cosolute(nu=nu, chi=chi, chiTS=chiTS)
        model = fh_crowding.BinaryCrowdingModel(
            protein=protein, cosolute=cosolute, eps=eps, epsTS=epsTS,
            dphiC=dphiC, phiC_max=phiC_max, T=T,
        )
    except MemoryError:
        st.error("**Memory Exhausted!** The grid resolution is too high for this deployment.")
        st.stop()
    except Exception as e:
        st.error(f"**Numerical Error initializing Binary Model:** {e}")
        st.stop()


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
    nu2     = st.sidebar.number_input("ν₂ (excluded volume)",          key="nu2",     step=0.01, min_value=1.0, format="%.4f")
    chi12   = st.sidebar.number_input("χ₁₂ (non-ideal mixing)",          key="chi12",   step=0.01, format="%.4f")
    chiTS12 = st.sidebar.number_input("χₜₛ₁₂ (entropy component)",      key="chiTS12", step=0.01, format="%.4f")
    
    eps2    = st.sidebar.number_input("ε₂ (soft interaction)",           step=0.01, format="%.4f", key="tern_eps2_input")
    epsTS2  = st.sidebar.number_input("εTS₂ (entropy component of ε)",  step=0.01, format="%.4f", key="tern_epsts2_input")

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
    nu3     = st.sidebar.number_input("ν₃ (excluded volume)",          key="nu3",     step=0.01, min_value=1.0, format="%.4f")
    chi13   = st.sidebar.number_input("χ₁₃ (non-ideal mixing)",          key="chi13",   step=0.01, format="%.4f")
    chiTS13 = st.sidebar.number_input("χₜₛ₁₃ (entropy component)",      key="chiTS13", step=0.01, format="%.4f")
    
    eps3    = st.sidebar.number_input("ε₃ (soft interaction)",           step=0.01, format="%.4f", key="tern_eps3_input")
    epsTS3  = st.sidebar.number_input("εTS₃ (entropy component of ε)",  step=0.01, format="%.4f", key="tern_epsts3_input")

    # --- Cosolute–cosolute non-ideal mixing ---
    st.sidebar.subheader("Cosolute–Cosolute Mixing (χ₂₃)")
    chi23   = st.sidebar.number_input("χ₂₃ (non-ideal mixing)",       key="chi23",   step=0.01, format="%.4f")
    chiTS23 = st.sidebar.number_input("χₜₛ₂₃ (entropy component)",   key="chiTS23", step=0.01, format="%.4f")

    # --- Synergy parameter (three-body coupling) ---
    st.sidebar.subheader("Synergy Parameter (ε₂₃)")
    eps23   = st.sidebar.number_input("ε₂₃ (synergy)",                step=0.01, format="%.4f", key="eps23")
    epsTS23 = st.sidebar.number_input("εTS₂₃ (entropy component)",   step=0.01, format="%.4f", key="epsTS23")

    # --- Concentration grid ---
    st.sidebar.subheader("Simulation Grid")
    
    tern_advanced = st.sidebar.checkbox(
        "Enable Advanced/High-Resolution Grid",
        value=False,
        key="tern_advanced_grid",
        help="Allows extremely fine grids. Warning: may crash the app if memory limits are exceeded."
    )
    
    tern_min_dphi = 1e-5 if tern_advanced else 0.001
    
    dphi2 = st.sidebar.number_input(
        "Δϕ₂ (grid step)",
        min_value=tern_min_dphi,
        max_value=0.05,
        value=0.01,
        step=0.0005,
        format="%.5f",
        key="tern_dphi2",
        help="Grid step for cosolute 2 axis. Package default: 0.0001. App default: 0.001.",
    )
    dphi3 = st.sidebar.number_input(
        "Δϕ₃ (grid step)",
        min_value=tern_min_dphi,
        max_value=0.05,
        value=0.01,
        step=0.0005,
        format="%.5f",
        key="tern_dphi3",
        help="Grid step for cosolute 3 axis. Package default: 0.0001. App default: 0.001.",
    )
    phi2_max = st.sidebar.number_input(
        "ϕ₂ max",
        min_value=0.001,
        max_value=1.0,
        value=0.2,
        step=0.01,
        format="%.3f",
        key="tern_phi2_max",
        help="Maximum volume fraction of cosolute 2. Default: 0.2.",
    )
    phi3_max = st.sidebar.number_input(
        "ϕ₃ max",
        min_value=0.001,
        max_value=1.0,
        value=0.2,
        step=0.01,
        format="%.3f",
        key="tern_phi3_max",
        help="Maximum volume fraction of cosolute 3. Default: 0.2.",
    )

    validation.check_ternary_composition(phi2_max, phi3_max)
    
    n_points = validation.estimate_ternary_grid_points(0.0001, phi2_max, dphi2, 0.0001, phi3_max, dphi3)
    n_points, mem_mb = validation.validate_grid_size(n_points, tern_advanced, num_arrays=10)
    
    if dphi2 > 0 and dphi3 > 0:
        st.sidebar.caption(f"Grid: {int(phi2_max/dphi2)}x{int(phi3_max/dphi3)} ({n_points:,} pts | ~{mem_mb:.1f} MB)")

    try:
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
    except MemoryError:
        st.error("**Memory Exhausted!** The grid resolution is too high for this deployment.")
        st.stop()
    except Exception as e:
        st.error(f"**Numerical Error initializing Ternary Model:** {e}")
        st.stop()

    citation.render_sidebar_citation()

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
styles.section_header("Experimental Data & Fitting Hub", "📋")
with st.expander("📊 Upload Experimental Data & Unit Settings (Optional)", expanded=False):
    st.markdown(
        "Upload your experimental CSV files to fit soft interaction parameters "
        "($\\varepsilon$, $\\varepsilon_{TS}$) and overlay data on model plots."
    )
    
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
        st.caption("🔒 **Data Privacy:** Uploaded files are processed during the app session and are not stored by the app. Do not upload confidential data unless you are comfortable using this deployment environment.")
        if model_type == "Binary Crowding Model":
            st.caption(
                "**Binary CSV format:** columns concentration (in the unit selected on the left), "
                "and at least one thermodynamic property column (e.g., ΔΔG, ΔΔH, TΔΔS) in kJ/mol or kcal/mol. "
                "Optional error columns: err_ΔΔG, err_ΔΔH, err_TΔΔS."
            )
            upload_mode = st.radio(
                "Data Upload Format",
                ["Single CSV File (concentration, ΔG, ΔH, TΔS)", "Separate CSV Files for ΔG, ΔH, TΔS"],
                key="bin_upload_mode",
                horizontal=True
            )
            
            if upload_mode == "Single CSV File (concentration, ΔG, ΔH, TΔS)":
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
                                df.loc[df["concentration"] <= 0.0, "concentration"] = 0.0001
                                
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
                    f_G = st.file_uploader("Upload ΔG CSV (concentration, ΔG)", type=["csv"], key="bin_g_uploader")
                with col_f2:
                    f_H = st.file_uploader("Upload ΔH CSV (concentration, ΔH)", type=["csv"], key="bin_h_uploader")
                with col_f3:
                    f_S = st.file_uploader("Upload TΔS CSV (concentration, TΔS)", type=["csv"], key="bin_s_uploader")
                
                if f_G:
                    try:
                        df = read_uploaded_csv(f_G)
                        if "concentration" not in df.columns or "dG" not in df.columns:
                            st.error(f"Missing required columns for ΔG. Expected: 'concentration', 'dG'.\n\nDetected columns: {list(df.columns)}")
                        else:
                            df["concentration"] = pd.to_numeric(df["concentration"], errors='coerce')
                            df["dG"] = pd.to_numeric(df["dG"], errors='coerce')
                            df = df.dropna(subset=["concentration", "dG"])
                            df.loc[df["concentration"] <= 0.0, "concentration"] = 0.0001
                            st.session_state["exp_conc_G"] = df["concentration"].values
                            st.session_state["raw_exp_ddG"] = df["dG"].values
                            st.session_state["exp_ddG"] = df["dG"].values * energy_mult
                            st.session_state["raw_err_ddG"] = df["err_dG"].values if "err_dG" in df.columns else np.nan
                            st.session_state["err_ddG"] = (df["err_dG"].values * energy_mult) if "err_dG" in df.columns else np.nan
                            st.session_state["exp_data_loaded"] = True
                            st.success("ΔG experimental data loaded!")
                    except Exception as ex:
                        st.error(f"Error reading ΔG CSV file: {ex}")
                        
                if f_H:
                    try:
                        df = read_uploaded_csv(f_H)
                        if "concentration" not in df.columns or "dH" not in df.columns:
                            st.error(f"Missing required columns for ΔH. Expected: 'concentration', 'dH'.\n\nDetected columns: {list(df.columns)}")
                        else:
                            df["concentration"] = pd.to_numeric(df["concentration"], errors='coerce')
                            df["dH"] = pd.to_numeric(df["dH"], errors='coerce')
                            df = df.dropna(subset=["concentration", "dH"])
                            df.loc[df["concentration"] <= 0.0, "concentration"] = 0.0001
                            st.session_state["exp_conc_T"] = df["concentration"].values
                            st.session_state["raw_exp_ddH"] = df["dH"].values
                            st.session_state["exp_ddH"] = df["dH"].values * energy_mult
                            st.session_state["raw_err_ddH"] = df["err_dH"].values if "err_dH" in df.columns else np.nan
                            st.session_state["err_ddH"] = (df["err_dH"].values * energy_mult) if "err_dH" in df.columns else np.nan
                            st.session_state["exp_data_loaded"] = True
                            st.success("ΔH experimental data loaded!")
                    except Exception as ex:
                        st.error(f"Error reading ΔH CSV file: {ex}")
                        
                if f_S:
                    try:
                        df = read_uploaded_csv(f_S)
                        if "concentration" not in df.columns or "TdS" not in df.columns:
                            st.error(f"Missing required columns for TΔS. Expected: 'concentration', 'TdS'.\n\nDetected columns: {list(df.columns)}")
                        else:
                            df["concentration"] = pd.to_numeric(df["concentration"], errors='coerce')
                            df["TdS"] = pd.to_numeric(df["TdS"], errors='coerce')
                            df = df.dropna(subset=["concentration", "TdS"])
                            df.loc[df["concentration"] <= 0.0, "concentration"] = 0.0001
                            st.session_state["exp_conc_T"] = df["concentration"].values
                            st.session_state["raw_exp_TddS"] = df["TdS"].values
                            st.session_state["exp_TddS"] = df["TdS"].values * energy_mult
                            st.session_state["raw_err_TddS"] = df["err_TdS"].values if "err_TdS" in df.columns else np.nan
                            st.session_state["err_TddS"] = (df["err_TdS"].values * energy_mult) if "err_TdS" in df.columns else np.nan
                            st.session_state["exp_data_loaded"] = True
                            st.success("TΔS experimental data loaded!")
                    except Exception as ex:
                        st.error(f"Error reading TΔS CSV file: {ex}")
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
                    f_G = st.file_uploader("Upload ΔG CSV (conc2, conc3, ΔG)", type=["csv"], key="tern_g_uploader")
                with col_tf2:
                    f_H = st.file_uploader("Upload ΔH CSV (conc2, conc3, ΔH)", type=["csv"], key="tern_h_uploader")
                with col_tf3:
                    f_S = st.file_uploader("Upload TΔS CSV (conc2, conc3, TΔS)", type=["csv"], key="tern_s_uploader")
                
                if f_G:
                    try:
                        df = read_uploaded_csv(f_G)
                        if "conc2" not in df.columns or "conc3" not in df.columns or "dG" not in df.columns:
                            st.error(f"Missing required columns for ternary ΔG. Expected: 'conc2', 'conc3', 'dG'.\n\nDetected columns: {list(df.columns)}")
                        else:
                            df["conc2"] = pd.to_numeric(df["conc2"], errors='coerce')
                            df["conc3"] = pd.to_numeric(df["conc3"], errors='coerce')
                            df["dG"] = pd.to_numeric(df["dG"], errors='coerce')
                            df = df.dropna(subset=["conc2", "conc3", "dG"])
                            df.loc[df["conc2"] <= 0.0, "conc2"] = 0.0001
                            df.loc[df["conc3"] <= 0.0, "conc3"] = 0.0001
                            st.session_state["exp_conc2"] = df["conc2"].values
                            st.session_state["exp_conc3"] = df["conc3"].values
                            st.session_state["exp_conc2_G"] = df["conc2"].values
                            st.session_state["exp_conc3_G"] = df["conc3"].values
                            st.session_state["raw_exp_val_G"] = df["dG"].values
                            st.session_state["exp_val_G"] = df["dG"].values * energy_mult
                            st.session_state["exp_data_loaded"] = True
                            st.success("Ternary ΔG data loaded!")
                    except Exception as ex:
                        st.error(f"Error reading ΔG CSV file: {ex}")
                if f_H:
                    try:
                        df = read_uploaded_csv(f_H)
                        if "conc2" not in df.columns or "conc3" not in df.columns or "dH" not in df.columns:
                            st.error(f"Missing required columns for ternary ΔH. Expected: 'conc2', 'conc3', 'dH'.\n\nDetected columns: {list(df.columns)}")
                        else:
                            df["conc2"] = pd.to_numeric(df["conc2"], errors='coerce')
                            df["conc3"] = pd.to_numeric(df["conc3"], errors='coerce')
                            df["dH"] = pd.to_numeric(df["dH"], errors='coerce')
                            df = df.dropna(subset=["conc2", "conc3", "dH"])
                            df.loc[df["conc2"] <= 0.0, "conc2"] = 0.0001
                            df.loc[df["conc3"] <= 0.0, "conc3"] = 0.0001
                            st.session_state["exp_conc2"] = df["conc2"].values
                            st.session_state["exp_conc3"] = df["conc3"].values
                            st.session_state["exp_conc2_T"] = df["conc2"].values
                            st.session_state["exp_conc3_T"] = df["conc3"].values
                            st.session_state["raw_exp_val_H"] = df["dH"].values
                            st.session_state["exp_val_H"] = df["dH"].values * energy_mult
                            st.session_state["exp_data_loaded"] = True
                            st.success("Ternary ΔH data loaded!")
                    except Exception as ex:
                        st.error(f"Error reading ΔH CSV file: {ex}")
                if f_S:
                    try:
                        df = read_uploaded_csv(f_S)
                        if "conc2" not in df.columns or "conc3" not in df.columns or "TdS" not in df.columns:
                            st.error(f"Missing required columns for ternary TΔS. Expected: 'conc2', 'conc3', 'TdS'.\n\nDetected columns: {list(df.columns)}")
                        else:
                            df["conc2"] = pd.to_numeric(df["conc2"], errors='coerce')
                            df["conc3"] = pd.to_numeric(df["conc3"], errors='coerce')
                            df["TdS"] = pd.to_numeric(df["TdS"], errors='coerce')
                            df = df.dropna(subset=["conc2", "conc3", "TdS"])
                            df.loc[df["conc2"] <= 0.0, "conc2"] = 0.0001
                            df.loc[df["conc3"] <= 0.0, "conc3"] = 0.0001
                            st.session_state["exp_conc2"] = df["conc2"].values
                            st.session_state["exp_conc3"] = df["conc3"].values
                            st.session_state["exp_conc2_T"] = df["conc2"].values
                            st.session_state["exp_conc3_T"] = df["conc3"].values
                            st.session_state["raw_exp_val_S"] = df["TdS"].values
                            st.session_state["exp_val_S"] = df["TdS"].values * energy_mult
                            st.session_state["exp_data_loaded"] = True
                            st.success("Ternary TΔS data loaded!")
                    except Exception as ex:
                        st.error(f"Error reading TΔS CSV file: {ex}")
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
            st.session_state["is_fitting_mode"] = False
            
        except Exception as e:
            st.error(f"Error during simulation: {e}")

    # Display solved model results and download if available
    if "solved_model" in st.session_state and st.session_state["solved_model_type"] == model_type:
        solved_model = st.session_state["solved_model"]
        
        st.markdown("### Export Simulated Quantities")
        
        # Lazy CSV generation to prevent memory spikes on every slider change
        current_model_id = id(solved_model)
        if st.session_state.get("csv_model_id") != current_model_id:
            st.session_state["csv_model_id"] = current_model_id
            st.session_state["csv_generated"] = False
            st.session_state["csv_data"] = ""
            
        if not st.session_state.get("csv_generated"):
            if st.button("🔄 Generate Full Results for Download", use_container_width=True, key="btn_gen_csv"):
                with st.spinner("Generating CSV..."):
                    st.session_state["csv_data"] = export.get_model_results_csv(solved_model)
                    st.session_state["csv_generated"] = True
                    st.rerun()
        else:
            filename_prefix = "fh_crowding_binary" if model_type == "Binary Crowding Model" else "fh_crowding_ternary"
            st.download_button(
                "📥 Download Simulation Results (CSV)",
                data=st.session_state["csv_data"],
                file_name=f"{filename_prefix}_model_results.csv",
                mime="text/csv",
                use_container_width=True,
                key="btn_download_csv"
            )


with col_fit:
    st.subheader("🛠️ Parameter Fitting")
    st.markdown(
        "Fit soft interaction parameters ($\\varepsilon$, $\\varepsilon_{TS}$) "
        "to your uploaded experimental dataset."
    )
    
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
                st.markdown("**Free Energy**")
                fit_eps_btn = st.button("Fit ε (from ΔΔG)", key="btn_fit_eps", use_container_width=True)
                if fit_eps_btn:
                    if st.session_state.get("exp_ddG") is not None and st.session_state.get("exp_conc_G") is not None:
                        fit_progress = st.progress(0, text="Fitting ε...")
                        try:
                            # Filter NaNs
                            conc_G = np.array(st.session_state["exp_conc_G"])
                            ddG = np.array(st.session_state["exp_ddG"])
                            valid_idx = np.isfinite(conc_G) & np.isfinite(ddG)
                            if not np.any(valid_idx):
                                raise ValueError("No valid non-NaN experimental ΔG data points to fit.")
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
                            st.session_state["is_fitting_mode"] = True
                            
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
                        st.error("Please upload experimental ΔG data first!")
                        
            # Button 2: Fit epsTS
            with col_b2:
                st.markdown("**Entropy & Enthalpy**")
                fit_epsts_btn = st.button("Fit εTS (from ΔΔH, TΔΔS)", key="btn_fit_epsts", use_container_width=True)
                if fit_epsts_btn:
                    if (st.session_state.get("exp_ddH") is not None and 
                        st.session_state.get("exp_TddS") is not None and 
                        st.session_state.get("exp_conc_T") is not None):
                        fit_progress = st.progress(0, text="Fitting εTS...")
                        try:
                            # Filter NaNs
                            conc_T = np.array(st.session_state["exp_conc_T"])
                            ddH = np.array(st.session_state["exp_ddH"])
                            TddS = np.array(st.session_state["exp_TddS"])
                            valid_idx = np.isfinite(conc_T) & np.isfinite(ddH) & np.isfinite(TddS)
                            if not np.any(valid_idx):
                                raise ValueError("No valid non-NaN experimental ΔH/TΔS data points to fit.")
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
                            st.session_state["is_fitting_mode"] = True
                            
                            st.session_state["fit_success_msg"] = f"Successfully fitted εTS: {model.epsTS:.4f}"
                            if hasattr(model, "resTS") and hasattr(model.resTS, "success") and not model.resTS.success:
                                msg = getattr(model.resTS, "message", "Optimizer did not converge.")
                                st.session_state["fit_warning_msg"] = f"⚠️ Optimization Warning: {msg}"
                            else:
                                st.session_state["fit_warning_msg"] = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Fitting error: {e}")
                    else:
                        st.error("Please upload experimental ΔH and TΔS data first!")
                        
            # Display current fitted values
            st.markdown("### Fitted Parameters")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                eps_val = f"{model.eps:.4f}" if st.session_state["fitted_eps"] is not None else "—"
                st.markdown(
                    styles.param_card("ε", eps_val, "Soft interaction (free energy component)"),
                    unsafe_allow_html=True
                )
            with col_m2:
                epsts_val = f"{model.epsTS:.4f}" if st.session_state["fitted_epsTS"] is not None else "—"
                st.markdown(
                    styles.param_card("εTS", epsts_val, "Soft interaction (entropy component)"),
                    unsafe_allow_html=True
                )
            
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
            
            # Column 1: Fit eps
            with col_t1:
                st.markdown("**Free Energy**")
                
                # Fit eps2
                fit_eps2_btn = st.button("Fit ε₂ (φ₃ = 0)", key="btn_fit_eps2", use_container_width=True)
                if fit_eps2_btn:
                    if (st.session_state.get("exp_val_G") is not None and 
                        st.session_state.get("exp_conc2_G") is not None and 
                        st.session_state.get("exp_conc3_G") is not None):
                        fit_progress = st.progress(0, text="Fitting ε₂ (where φ₃ = 0)...")
                        try:
                            conc2 = np.array(st.session_state["exp_conc2_G"])
                            conc3 = np.array(st.session_state["exp_conc3_G"])
                            val_G = np.array(st.session_state["exp_val_G"])
                            if len(conc2) != len(val_G) or len(conc3) != len(val_G):
                                raise ValueError(f"Mismatched shapes: concentration arrays have lengths {len(conc2)} and {len(conc3)}, but dG has length {len(val_G)}.")
                            valid_idx = np.isfinite(conc2) & np.isfinite(conc3) & np.isfinite(val_G) & (conc3 <= 0.0011)
                            if not np.any(valid_idx):
                                raise ValueError("No valid non-NaN experimental ΔG data points with phi3 <= 0.0011 to fit eps2.")
                            
                            model.fit_eps(
                                conc2[valid_idx],
                                conc3[valid_idx],
                                val_G[valid_idx],
                                concentration_type=fit_conc_type,
                                fit_eps2=True,
                                fit_eps3=False,
                                fit_eps23=False
                            )
                            fit_progress.progress(0.5, text="Solving equilibrium with fitted eps2 parameter...")
                            model.solve_equil(print_msg=False)
                            model.to_pandas()
                            fit_progress.progress(1.0, text="Fit & Simulation updated!")
                            
                            st.session_state["fitted_eps2"] = model.eps2
                            st.session_state["fit_updated"] = True
                            st.session_state["solved_model"] = model
                            st.session_state["solved_model_type"] = model_type
                            st.session_state["is_fitting_mode"] = True
                            st.session_state["fit_success_msg"] = f"Successfully fitted ε₂: {model.eps2:.4f}"
                            if hasattr(model, "res") and hasattr(model.res, "success") and not model.res.success:
                                msg = getattr(model.res, "message", "Optimizer did not converge.")
                                st.session_state["fit_warning_msg"] = f"⚠️ Optimization Warning: {msg}"
                            else:
                                st.session_state["fit_warning_msg"] = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Fitting error: {e}")
                    else:
                        st.error("Please upload experimental Ternary ΔG data first!")
                
                # Fit eps3
                fit_eps3_btn = st.button("Fit ε₃ (φ₂ = 0)", key="btn_fit_eps3", use_container_width=True)
                if fit_eps3_btn:
                    if (st.session_state.get("exp_val_G") is not None and 
                        st.session_state.get("exp_conc2_G") is not None and 
                        st.session_state.get("exp_conc3_G") is not None):
                        fit_progress = st.progress(0, text="Fitting ε₃ (where φ₂ = 0)...")
                        try:
                            conc2 = np.array(st.session_state["exp_conc2_G"])
                            conc3 = np.array(st.session_state["exp_conc3_G"])
                            val_G = np.array(st.session_state["exp_val_G"])
                            if len(conc2) != len(val_G) or len(conc3) != len(val_G):
                                raise ValueError(f"Mismatched shapes: concentration arrays have lengths {len(conc2)} and {len(conc3)}, but dG has length {len(val_G)}.")
                            valid_idx = np.isfinite(conc2) & np.isfinite(conc3) & np.isfinite(val_G) & (conc2 <= 0.0011)
                            if not np.any(valid_idx):
                                raise ValueError("No valid non-NaN experimental ΔG data points with phi2 <= 0.0011 to fit eps3.")
                            
                            model.fit_eps(
                                conc2[valid_idx],
                                conc3[valid_idx],
                                val_G[valid_idx],
                                concentration_type=fit_conc_type,
                                fit_eps2=False,
                                fit_eps3=True,
                                fit_eps23=False
                            )
                            fit_progress.progress(0.5, text="Solving equilibrium with fitted eps3 parameter...")
                            model.solve_equil(print_msg=False)
                            model.to_pandas()
                            fit_progress.progress(1.0, text="Fit & Simulation updated!")
                            
                            st.session_state["fitted_eps3"] = model.eps3
                            st.session_state["fit_updated"] = True
                            st.session_state["solved_model"] = model
                            st.session_state["solved_model_type"] = model_type
                            st.session_state["is_fitting_mode"] = True
                            st.session_state["fit_success_msg"] = f"Successfully fitted ε₃: {model.eps3:.4f}"
                            if hasattr(model, "res") and hasattr(model.res, "success") and not model.res.success:
                                msg = getattr(model.res, "message", "Optimizer did not converge.")
                                st.session_state["fit_warning_msg"] = f"⚠️ Optimization Warning: {msg}"
                            else:
                                st.session_state["fit_warning_msg"] = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Fitting error: {e}")
                    else:
                        st.error("Please upload experimental Ternary ΔG data first!")
 
                # Fit eps23
                eps23_enabled = (st.session_state.get("fitted_eps2") is not None and 
                                 st.session_state.get("fitted_eps3") is not None)
                fit_eps23_btn = st.button(
                    "Fit ε₂₃ (All data)", 
                    key="btn_fit_eps23", 
                    use_container_width=True,
                    disabled=not eps23_enabled,
                    help="Only active after ε₂ and ε₃ are successfully fitted."
                )
                if fit_eps23_btn:
                    if (st.session_state.get("exp_val_G") is not None and 
                        st.session_state.get("exp_conc2_G") is not None and 
                        st.session_state.get("exp_conc3_G") is not None):
                        fit_progress = st.progress(0, text="Fitting ε₂₃ (using all data)...")
                        try:
                            conc2 = np.array(st.session_state["exp_conc2_G"])
                            conc3 = np.array(st.session_state["exp_conc3_G"])
                            val_G = np.array(st.session_state["exp_val_G"])
                            if len(conc2) != len(val_G) or len(conc3) != len(val_G):
                                raise ValueError(f"Mismatched shapes: concentration arrays have lengths {len(conc2)} and {len(conc3)}, but dG has length {len(val_G)}.")
                            valid_idx = np.isfinite(conc2) & np.isfinite(conc3) & np.isfinite(val_G)
                            if not np.any(valid_idx):
                                raise ValueError("No valid non-NaN experimental ΔG data points to fit eps23.")
                            
                            # Keep previously fitted eps2 and eps3 constant
                            model.eps2 = st.session_state["fitted_eps2"]
                            model.eps3 = st.session_state["fitted_eps3"]
                            
                            model.fit_eps(
                                conc2[valid_idx],
                                conc3[valid_idx],
                                val_G[valid_idx],
                                concentration_type=fit_conc_type,
                                fit_eps2=False,
                                fit_eps3=False,
                                fit_eps23=True
                            )
                            fit_progress.progress(0.5, text="Solving equilibrium with fitted eps23 synergy parameter...")
                            model.solve_equil(print_msg=False)
                            model.to_pandas()
                            fit_progress.progress(1.0, text="Fit & Simulation updated!")
                            
                            st.session_state["fitted_eps23"] = model.eps23
                            st.session_state["fit_updated"] = True
                            st.session_state["solved_model"] = model
                            st.session_state["solved_model_type"] = model_type
                            st.session_state["is_fitting_mode"] = True
                            st.session_state["fit_success_msg"] = f"Successfully fitted ε₂₃: {model.eps23:.4f}"
                            if hasattr(model, "res") and hasattr(model.res, "success") and not model.res.success:
                                msg = getattr(model.res, "message", "Optimizer did not converge.")
                                st.session_state["fit_warning_msg"] = f"⚠️ Optimization Warning: {msg}"
                            else:
                                st.session_state["fit_warning_msg"] = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Fitting error: {e}")
                    else:
                        st.error("Please upload experimental Ternary ΔG data first!")
            
            # Column 2: Fit epsTS
            with col_t2:
                st.markdown("**Entroppy & Enthalpy**")
                
                # Fit epsTS2
                fit_epsts2_btn = st.button("Fit εTS₂ (φ₃ = 0)", key="btn_fit_epsts2", use_container_width=True)
                if fit_epsts2_btn:
                    if (st.session_state.get("exp_val_H") is not None and 
                        st.session_state.get("exp_val_S") is not None and 
                        st.session_state.get("exp_conc2_T") is not None and 
                        st.session_state.get("exp_conc3_T") is not None):
                        fit_progress = st.progress(0, text="Fitting εTS₂ (where φ₃ = 0)...")
                        try:
                            conc2 = np.array(st.session_state["exp_conc2_T"])
                            conc3 = np.array(st.session_state["exp_conc3_T"])
                            val_H = np.array(st.session_state["exp_val_H"])
                            val_S = np.array(st.session_state["exp_val_S"])
                            if len(conc2) != len(val_H) or len(conc3) != len(val_H) or len(val_S) != len(val_H):
                                raise ValueError(f"Mismatched shapes: concentration arrays have lengths {len(conc2)} and {len(conc3)}, dH has length {len(val_H)}, and TdS has length {len(val_S)}.")
                            valid_idx = np.isfinite(conc2) & np.isfinite(conc3) & np.isfinite(val_H) & np.isfinite(val_S) & (conc3 <= 0.0011)
                            if not np.any(valid_idx):
                                raise ValueError("No valid non-NaN experimental ΔH/TΔS data points with phi3 <= 0.0011 to fit epsTS2.")
                            
                            model.fit_epsTS(
                                conc2[valid_idx],
                                conc3[valid_idx],
                                val_H[valid_idx],
                                val_S[valid_idx],
                                concentration_type=fit_conc_type,
                                fit_epsTS2=True,
                                fit_epsTS3=False,
                                fit_epsTS23=False
                            )
                            fit_progress.progress(0.5, text="Solving equilibrium with fitted epsTS2 parameter...")
                            model.solve_equil(print_msg=False)
                            model.to_pandas()
                            fit_progress.progress(1.0, text="Fit & Simulation updated!")
                            
                            st.session_state["fitted_epsTS2"] = model.epsTS2
                            st.session_state["fit_updated"] = True
                            st.session_state["solved_model"] = model
                            st.session_state["solved_model_type"] = model_type
                            st.session_state["is_fitting_mode"] = True
                            st.session_state["fit_success_msg"] = f"Successfully fitted εTS₂: {model.epsTS2:.4f}"
                            if hasattr(model, "resTS") and hasattr(model.resTS, "success") and not model.resTS.success:
                                msg = getattr(model.resTS, "message", "Optimizer did not converge.")
                                st.session_state["fit_warning_msg"] = f"⚠️ Optimization Warning: {msg}"
                            else:
                                st.session_state["fit_warning_msg"] = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Fitting error: {e}")
                    else:
                        st.error("Please upload experimental Ternary ΔH and TΔS data first!")

                # Fit epsTS3
                fit_epsts3_btn = st.button("Fit εTS₃ (φ₂ = 0)", key="btn_fit_epsts3", use_container_width=True)
                if fit_epsts3_btn:
                    if (st.session_state.get("exp_val_H") is not None and 
                        st.session_state.get("exp_val_S") is not None and 
                        st.session_state.get("exp_conc2_T") is not None and 
                        st.session_state.get("exp_conc3_T") is not None):
                        fit_progress = st.progress(0, text="Fitting εTS₃ (where φ₂ = 0)...")
                        try:
                            conc2 = np.array(st.session_state["exp_conc2_T"])
                            conc3 = np.array(st.session_state["exp_conc3_T"])
                            val_H = np.array(st.session_state["exp_val_H"])
                            val_S = np.array(st.session_state["exp_val_S"])
                            if len(conc2) != len(val_H) or len(conc3) != len(val_H) or len(val_S) != len(val_H):
                                raise ValueError(f"Mismatched shapes: concentration arrays have lengths {len(conc2)} and {len(conc3)}, dH has length {len(val_H)}, and TdS has length {len(val_S)}.")
                            valid_idx = np.isfinite(conc2) & np.isfinite(conc3) & np.isfinite(val_H) & np.isfinite(val_S) & (conc2 <= 0.0011)
                            if not np.any(valid_idx):
                                raise ValueError("No valid non-NaN experimental ΔH/TΔS data points with phi2 <= 0.0011 to fit epsTS3.")
                            
                            model.fit_epsTS(
                                conc2[valid_idx],
                                conc3[valid_idx],
                                val_H[valid_idx],
                                val_S[valid_idx],
                                concentration_type=fit_conc_type,
                                fit_epsTS2=False,
                                fit_epsTS3=True,
                                fit_epsTS23=False
                            )
                            fit_progress.progress(0.5, text="Solving equilibrium with fitted epsTS3 parameter...")
                            model.solve_equil(print_msg=False)
                            model.to_pandas()
                            fit_progress.progress(1.0, text="Fit & Simulation updated!")
                            
                            st.session_state["fitted_epsTS3"] = model.epsTS3
                            st.session_state["fit_updated"] = True
                            st.session_state["solved_model"] = model
                            st.session_state["solved_model_type"] = model_type
                            st.session_state["is_fitting_mode"] = True
                            st.session_state["fit_success_msg"] = f"Successfully fitted εTS₃: {model.epsTS3:.4f}"
                            if hasattr(model, "resTS") and hasattr(model.resTS, "success") and not model.resTS.success:
                                msg = getattr(model.resTS, "message", "Optimizer did not converge.")
                                st.session_state["fit_warning_msg"] = f"⚠️ Optimization Warning: {msg}"
                            else:
                                st.session_state["fit_warning_msg"] = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Fitting error: {e}")
                    else:
                        st.error("Please upload experimental Ternary ΔH and TΔS data first!")

                # Fit epsTS23
                epsts23_enabled = (st.session_state.get("fitted_epsTS2") is not None and 
                                   st.session_state.get("fitted_epsTS3") is not None)
                fit_epsts23_btn = st.button(
                    "Fit εTS₂₃ (All data)", 
                    key="btn_fit_epsts23", 
                    use_container_width=True,
                    disabled=not epsts23_enabled,
                    help="Only active after εTS₂ and εTS₃ are successfully fitted."
                )
                if fit_epsts23_btn:
                    if (st.session_state.get("exp_val_H") is not None and 
                        st.session_state.get("exp_val_S") is not None and 
                        st.session_state.get("exp_conc2_T") is not None and 
                        st.session_state.get("exp_conc3_T") is not None):
                        fit_progress = st.progress(0, text="Fitting εTS₂₃ (using all data)...")
                        try:
                            conc2 = np.array(st.session_state["exp_conc2_T"])
                            conc3 = np.array(st.session_state["exp_conc3_T"])
                            val_H = np.array(st.session_state["exp_val_H"])
                            val_S = np.array(st.session_state["exp_val_S"])
                            if len(conc2) != len(val_H) or len(conc3) != len(val_H) or len(val_S) != len(val_H):
                                raise ValueError(f"Mismatched shapes: concentration arrays have lengths {len(conc2)} and {len(conc3)}, dH has length {len(val_H)}, and TdS has length {len(val_S)}.")
                            valid_idx = np.isfinite(conc2) & np.isfinite(conc3) & np.isfinite(val_H) & np.isfinite(val_S)
                            if not np.any(valid_idx):
                                raise ValueError("No valid non-NaN experimental ΔH/TΔS data points to fit epsTS23.")
                            
                            # Keep previously fitted epsTS2 and epsTS3 constant
                            model.epsTS2 = st.session_state["fitted_epsTS2"]
                            model.epsTS3 = st.session_state["fitted_epsTS3"]
                            
                            model.fit_epsTS(
                                conc2[valid_idx],
                                conc3[valid_idx],
                                val_H[valid_idx],
                                val_S[valid_idx],
                                concentration_type=fit_conc_type,
                                fit_epsTS2=False,
                                fit_epsTS3=False,
                                fit_epsTS23=True
                            )
                            fit_progress.progress(0.5, text="Solving equilibrium with fitted epsTS23 synergy parameter...")
                            model.solve_equil(print_msg=False)
                            model.to_pandas()
                            fit_progress.progress(1.0, text="Fit & Simulation updated!")
                            
                            st.session_state["fitted_epsTS23"] = model.epsTS23
                            st.session_state["fit_updated"] = True
                            st.session_state["solved_model"] = model
                            st.session_state["solved_model_type"] = model_type
                            st.session_state["is_fitting_mode"] = True
                            st.session_state["fit_success_msg"] = f"Successfully fitted εTS₂₃: {model.epsTS23:.4f}"
                            if hasattr(model, "resTS") and hasattr(model.resTS, "success") and not model.resTS.success:
                                msg = getattr(model.resTS, "message", "Optimizer did not converge.")
                                st.session_state["fit_warning_msg"] = f"⚠️ Optimization Warning: {msg}"
                            else:
                                st.session_state["fit_warning_msg"] = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Fitting error: {e}")
                    else:
                        st.error("Please upload experimental Ternary ΔH and TΔS data first!")
                        
            # Display current fitted values
            st.markdown("### Fitted Parameters")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                eps2_val = f"{model.eps2:.4f}" if st.session_state["fitted_eps2"] is not None else "—"
                eps3_val = f"{model.eps3:.4f}" if st.session_state["fitted_eps3"] is not None else "—"
                eps23_val = f"{model.eps23:.4f}" if st.session_state["fitted_eps23"] is not None else "—"
                st.markdown(
                    styles.param_card("ε₂", eps2_val, "Cosolute 2 soft interaction") +
                    styles.param_card("ε₃", eps3_val, "Cosolute 3 soft interaction") +
                    styles.param_card("ε₂₃ (synergy)", eps23_val, "Soft interaction synergy coupling"),
                    unsafe_allow_html=True
                )
            with col_m2:
                epsts2_val = f"{model.epsTS2:.4f}" if st.session_state["fitted_epsTS2"] is not None else "—"
                epsts3_val = f"{model.epsTS3:.4f}" if st.session_state["fitted_epsTS3"] is not None else "—"
                epsts23_val = f"{model.epsTS23:.4f}" if st.session_state["fitted_epsTS23"] is not None else "—"
                st.markdown(
                    styles.param_card("εTS₂", epsts2_val, "Cosolute 2 entropy") +
                    styles.param_card("εTS₃", epsts3_val, "Cosolute 3 entropy") +
                    styles.param_card("εTS₂₃ (synergy)", epsts23_val, "Entropy synergy coupling"),
                    unsafe_allow_html=True
                )
                
            # Download fitted parameters button
            has_fit = (st.session_state["fitted_eps2"] is not None or 
                       st.session_state["fitted_eps3"] is not None or 
                       st.session_state["fitted_eps23"] is not None or 
                       st.session_state["fitted_epsTS2"] is not None or 
                       st.session_state["fitted_epsTS3"] is not None or 
                       st.session_state["fitted_epsTS23"] is not None)
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
    styles.section_header("Visualization & Plots", "📈")
    
    solved_model = st.session_state["solved_model"]
    
    # Checkbox to overlay experimental data (shown only if some exp data is uploaded)
    show_exp = False
    if st.session_state["exp_data_loaded"]:
        show_exp = st.checkbox("Overlay uploaded experimental data on plots", value=False, key="show_exp_data")
        
    plot_option = st.selectbox(
        "Select Plotting Mode",
        ["Standard Preset Plot", "Custom Axis Plot"],
        key="plot_mode_selectbox"
    )
    
    if plot_option == "Standard Preset Plot":
        if model_type == "Binary Crowding Model":
            st.subheader("Standard 3x3 Results Plot")
            col1, col2 = st.columns(2)
            with col1:
                conc_type_display = st.selectbox("Concentration axis type", ["φ", "molar", "molal"], key="bin_plot_conc")
                conc_type_plot = "phi" if conc_type_display == "φ" else conc_type_display
            with col2:
                plot_unit = st.selectbox("Plotting Unit", ["kJ/mol", "kcal/mol"], key="bin_plot_unit")
            
            # Setup experimental values to overlay if enabled
            folding_plot = (plot_unit == "kJ/mol")
            
            show_G = True
            show_HTS = True
            if st.session_state.get("is_fitting_mode", False):
                if st.session_state.get("fitted_eps") is None:
                    show_G = False
                if st.session_state.get("fitted_epsTS") is None:
                    show_HTS = False
            
            plot_kwargs = {"concentration_type": conc_type_plot, "folding": folding_plot, "show_G": show_G, "show_HTS": show_HTS}
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
            
            presets = [
                "ΔΔG (3x3 contour)",
                "φS (Contours of subdomain concentrations)",
                "Ms (Contours of subdomain volume fractions)",
                "μS2 (Contours of subdomain 2 chemical potentials)",
                "μS3 (Contours of subdomain 3 chemical potentials)",
                "TΔS_mix (Contours of mixing entropy)",
                "ΔG_mix (Contours of mixing free energy)",
                "ΔΔG_μ (Contours of ΔΔG chemical potentials)",
                "TΔΔS (Contours of TΔΔS entropy)",
                "ΔΔH (Contours of ΔΔH enthalpy)",
                "Γ (Contours of preferential interaction coefficients)",
                "Γ_μ (Contours of preferential interaction μ)"
            ]
            if st.session_state.get("is_fitting_mode", False):
                has_eps = (st.session_state.get("fitted_eps2") is not None and 
                           st.session_state.get("fitted_eps3") is not None and 
                           st.session_state.get("fitted_eps23") is not None)
                if not has_eps:
                    presets = ["Ms (Contours of subdomain volume fractions)"]
                elif st.session_state.get("fitted_epsTS23") is None:
                    presets = [p for p in presets if "TΔΔS" not in p and "ΔΔH" not in p and "TΔS_mix" not in p]
                    
            preset_plot = st.selectbox("Select Standard Plot", presets)
            plotter = fh_crowding.TernaryPlotter(solved_model)
            
            if "ΔΔG (3x3 contour)" in preset_plot:
                fig = plotter.plot_ddG()
            elif "φS" in preset_plot:
                fig = plotter.plot_phiS()
            elif "Ms" in preset_plot:
                fig = plotter.plot_Ms()
            elif "μS2" in preset_plot:
                fig = plotter.plot_mus2()
            elif "μS3" in preset_plot:
                fig = plotter.plot_mus3()
            elif "TΔS_mix" in preset_plot:
                fig = plotter.plot_TdS_mix()
            elif "ΔG_mix" in preset_plot:
                fig = plotter.plot_dG_mix()
            elif "ΔΔG_μ" in preset_plot:
                fig = plotter.plot_ddG_mu()
            elif "TΔΔS" in preset_plot:
                fig = plotter.plot_TddS()
            elif "ΔΔH" in preset_plot:
                fig = plotter.plot_ddH()
            elif "Γ_μ" in preset_plot:
                fig = plotter.plot_Gamma_mu()
            elif "Γ" in preset_plot:
                fig = plotter.plot_Gamma()
                
            # If ternary contour preset and show_exp is enabled, overlay exp points on the subplots
            # If ternary contour preset and show_exp is enabled, overlay exp points on the subplots
            is_fold_preset = any(p in preset_plot for p in ["ΔΔG (3x3 contour)", "TddS", "ddH"])
            if show_exp and is_fold_preset:
                is_G = "ΔΔG" in preset_plot
                c2_key = "exp_conc2_G" if is_G else "exp_conc2_T"
                c3_key = "exp_conc3_G" if is_G else "exp_conc3_T"
                if st.session_state.get(c2_key) is not None:
                    # Convert exp points concentration to phi
                    exp_x_phi = convert_exp_conc(st.session_state[c2_key], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=2)
                    exp_y_phi = convert_exp_conc(st.session_state[c3_key], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=3, exp_conc3=st.session_state[c3_key])
                    
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
                "Volume Fraction": ("phiC", r'$\phi_C$'),
                "Molar Concentration": ("molar", r'$\mathrm{Molar\ [M]}$'),
                "Molal Concentration": ("molal", r'$\mathrm{Molal\ [mol/kg]}$'),
                "Osmotic Pressure": ("osm", r'$\Pi\ \mathrm{(Osmolal)}$'),
                "Preferential Hydration": ("gamma", r'$\Delta\Gamma_S$'),
                "Preferential Interaction": ("gammaC", r'$\Delta\Gamma_C$'),
                "phiCsurf": ("phiCsurf", r'$\phi_C^{\mathrm{surf}}$'),
                "phiSsurf": ("phiSsurf", r'$\phi_S^{\mathrm{surf}}$'),
                "Free Energy [kJ]": ("ddA_kj", r'$\Delta\Delta G^{0}\ \mathrm{[kJ/mol]}$'),
                "Free Energy [kcal]": ("ddA_kcal", r'$\Delta\Delta G^{0}\ \mathrm{[kcal/mol]}$'),
                "Enthalpy [kJ]": ("ddE_kj", r'$\Delta\Delta H^{0}\ \mathrm{[kJ/mol]}$'),
                "Enthalpy [kcal]": ("ddE_kcal", r'$\Delta\Delta H^{0}\ \mathrm{[kcal/mol]}$'),
                "Entropy [kJ]": ("TddS_kj", r'$T\Delta\Delta S^{0}\ \mathrm{[kJ/mol]}$'),
                "Entropy [kcal]": ("TddS_kcal", r'$T\Delta\Delta S^{0}\ \mathrm{[kcal/mol]}$'),
            }
            
            if st.session_state.get("is_fitting_mode", False):
                if st.session_state.get("fitted_eps") is None:
                    properties = {k: v for k, v in properties.items() if "Free Energy" not in k}
                if st.session_state.get("fitted_epsTS") is None:
                    properties = {k: v for k, v in properties.items() if "Enthalpy" not in k and "Entropy" not in k}
            
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
                
            # ── Build Plotly figure ──────────────────────────────────────────
            pfig = go.Figure()
            x_data = getattr(solved_model, x_attr)

            if plot_contrib:
                # Helper: add a Plotly trace
                def _add_trace(yvals, name, color, dash="solid", width=2):
                    pfig.add_trace(go.Scatter(
                        x=x_data, y=yvals, mode="lines", name=name,
                        line=dict(color=color, width=width, dash=dash),
                        hovertemplate=f"{name}<br>x=%{{x:.4f}}<br>y=%{{y:.4f}}<extra></extra>"
                    ))

                if pot_type == "ddA":
                    suf = "_kj" if pot_unit == "kJ" else "_kcal"
                    _add_trace(getattr(solved_model, f"ddA{suf}"),    "Total ΔΔG",           styles.PLOT_TOTAL_COLOR, "solid", 2.5)
                    _add_trace(getattr(solved_model, f"ddA_nu{suf}"), "ν (Excluded Volume)", styles.PLOT_NU_COLOR,    "solid")
                    _add_trace(getattr(solved_model, f"ddA_chi{suf}"),"χ (Non-ideal mixing)",styles.PLOT_CHI_COLOR,   "solid")
                    _add_trace(getattr(solved_model, f"ddA_eps{suf}"),"ε (Soft interaction)",styles.PLOT_EPS_COLOR,   "solid")
                elif pot_type == "ddE":
                    suf = "_kj" if pot_unit == "kJ" else "_kcal"
                    _add_trace(getattr(solved_model, f"ddE{suf}"),    "Total ΔΔH",           styles.PLOT_TOTAL_COLOR, "solid", 2.5)
                    _add_trace(getattr(solved_model, f"ddE_chi{suf}"),"χ (Non-ideal mixing)",styles.PLOT_CHI_COLOR,   "solid")
                    _add_trace(getattr(solved_model, f"ddE_eps{suf}"),"ε (Soft interaction)",styles.PLOT_EPS_COLOR,   "solid")
                elif pot_type == "TddS":
                    suf = "_kj" if pot_unit == "kJ" else "_kcal"
                    _add_trace(getattr(solved_model, f"TddS{suf}"),    "Total TΔΔS",          styles.PLOT_TOTAL_COLOR, "solid", 2.5)
                    _add_trace(getattr(solved_model, f"TddS_nu{suf}"), "ν (Excluded Volume)", styles.PLOT_NU_COLOR,    "solid")
                    _add_trace(getattr(solved_model, f"TddS_chi{suf}"),"χ (Non-ideal mixing)",styles.PLOT_CHI_COLOR,   "solid")
                    _add_trace(getattr(solved_model, f"TddS_eps{suf}"),"ε (Soft interaction)",styles.PLOT_EPS_COLOR,   "solid")
            else:
                y_data = getattr(solved_model, y_attr)
                pfig.add_trace(go.Scatter(
                    x=x_data, y=y_data, mode="lines", name="Model",
                    line=dict(color=styles.PLOT_TOTAL_COLOR, width=2.5),
                    hovertemplate="Model<br>x=%{x:.4f}<br>y=%{y:.4f}<extra></extra>"
                ))

            # ── Experimental overlay ────────────────────────────────────────
            if show_exp:
                exp_x = None
                exp_y = None
                err_y = None
                y_conc = None
                
                if "ddA" in y_attr:
                    exp_y = st.session_state.get("exp_ddG")
                    err_y = st.session_state.get("err_ddG")
                    y_conc = st.session_state.get("exp_conc_G")
                    if "kcal" in y_attr and exp_y is not None:
                        exp_y = exp_y / 4.184
                        if err_y is not None: err_y = err_y / 4.184
                elif "ddE" in y_attr:
                    exp_y = st.session_state.get("exp_ddH")
                    err_y = st.session_state.get("err_ddH")
                    y_conc = st.session_state.get("exp_conc_T")
                    if "kcal" in y_attr and exp_y is not None:
                        exp_y = exp_y / 4.184
                        if err_y is not None: err_y = err_y / 4.184
                elif "TddS" in y_attr:
                    exp_y = st.session_state.get("exp_TddS")
                    err_y = st.session_state.get("err_TddS")
                    y_conc = st.session_state.get("exp_conc_T")
                    if "kcal" in y_attr and exp_y is not None:
                        exp_y = exp_y / 4.184
                        if err_y is not None: err_y = err_y / 4.184
                        
                if exp_y is not None and y_conc is not None:
                    if x_attr in ["phiC", "molar", "molal"]:
                        exp_x = convert_exp_conc(
                            y_conc, from_type=uploaded_conc_unit,
                            to_type=x_attr if x_attr != "phi" else "phiC",
                            model=solved_model
                        )
                    elif "ddA" in x_attr:
                        ref_g = st.session_state.get("exp_conc_G")
                        val_g = st.session_state.get("exp_ddG")
                        if ref_g is not None and val_g is not None:
                            sort_idx = np.argsort(ref_g)
                            exp_x = np.interp(y_conc, ref_g[sort_idx], val_g[sort_idx])
                            if "kcal" in x_attr: exp_x = exp_x / 4.184
                    elif "ddE" in x_attr:
                        ref_t = st.session_state.get("exp_conc_T")
                        val_h = st.session_state.get("exp_ddH")
                        if ref_t is not None and val_h is not None:
                            sort_idx = np.argsort(ref_t)
                            exp_x = np.interp(y_conc, ref_t[sort_idx], val_h[sort_idx])
                            if "kcal" in x_attr: exp_x = exp_x / 4.184
                    elif "TddS" in x_attr:
                        ref_t = st.session_state.get("exp_conc_T")
                        val_s = st.session_state.get("exp_TddS")
                        if ref_t is not None and val_s is not None:
                            sort_idx = np.argsort(ref_t)
                            exp_x = np.interp(y_conc, ref_t[sort_idx], val_s[sort_idx])
                            if "kcal" in x_attr: exp_x = exp_x / 4.184
                            
                if exp_y is not None and exp_x is not None:
                    has_err = err_y is not None and not np.all(np.isnan(
                        np.array(err_y, dtype=float) if not isinstance(err_y, float) else [err_y]
                    ))
                    pfig.add_trace(go.Scatter(
                        x=exp_x, y=exp_y, mode="markers", name="Experimental",
                        marker=dict(
                            color=styles.PLOT_EXP_COLOR, size=10,
                            symbol="circle", line=dict(color="black", width=1.2)
                        ),
                        error_y=dict(
                            type="data", array=list(err_y) if has_err else None, visible=has_err,
                            color=styles.PLOT_EXP_COLOR, thickness=1.5, width=4
                        ) if has_err else None,
                        hovertemplate="Experimental<br>x=%{x:.4f}<br>y=%{y:.4f}<extra></extra>"
                    ))

            pfig.update_layout(
                xaxis_title=x_label,
                yaxis_title=y_label,
                plot_bgcolor="white",
                paper_bgcolor="white",
                legend=dict(
                    orientation="v", x=1.02, xanchor="left",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#dde3e8", borderwidth=1
                ),
                margin=dict(l=60, r=20, t=30, b=60),
                font=dict(family="Inter, DejaVu Sans", size=12),
                xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor=styles.PALETTE_DARK, linewidth=1, mirror=True),
                yaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor=styles.PALETTE_DARK, linewidth=1, mirror=True),
            )
            try:
                _display_and_export_plotly(pfig, "fh_crowding_binary_custom_plot", "bin_custom_plot")
            except Exception as e:
                st.error(f"Error rendering custom 1D plot: {e}")
            
        else: # Ternary Custom Plot
            st.subheader("Custom Ternary Plotting")
            
            tern_mode = st.radio("Plot Type", ["2D Contour Plot", "1D Slice Plot", "3D Surface Plot"])
            
            # Dual-label system: (attr, latex_label, unicode_label)
            # latex_label → matplotlib (contour, slice) where LaTeX renders natively
            # unicode_label → 3D Plotly (WebGL) where MathJax cannot render
            properties_contour = {
                "Free Energy (ΔΔG) [kT]": ("ddG", r'$\Delta\Delta G^{0} / (k_B T)$', "ΔΔG⁰ / (k_B T)"),
                "Free Energy (ΔΔG) [kJ]": ("ddG_kJ", r'$\Delta\Delta G^{0}\ \mathrm{[kJ/mol]}$', "ΔΔG⁰ [kJ/mol]"),
                "Enthalpy (ΔΔH) [kT]": ("ddH", r'$\Delta\Delta H^{0} / (k_B T)$', "ΔΔH⁰ / (k_B T)"),
                "Enthalpy (ΔΔH) [kJ]": ("ddH_kJ", r'$\Delta\Delta H^{0}\ \mathrm{[kJ/mol]}$', "ΔΔH⁰ [kJ/mol]"),
                "Entropy (TΔΔS) [kT]": ("TddS", r'$T\Delta\Delta S^{0} / (k_B T)$', "TΔΔS⁰ / (k_B T)"),
                "Entropy (TΔΔS) [kJ]": ("TddS_kJ", r'$T\Delta\Delta S^{0}\ \mathrm{[kJ/mol]}$', "TΔΔS⁰ [kJ/mol]"),
                "Osmotic Pressure": ("osm", r'$\Pi\ \mathrm{(Osmolal)}$', "Π (Osmolal)"),
                "Preferential Interaction 2 (Γ₂)": ("Gamma_2", r'$\Delta\Gamma_2$', "ΔΓ₂"),
                "Preferential Interaction 3 (Γ₃)": ("Gamma_3", r'$\Delta\Gamma_3$', "ΔΓ₃"),
                "Preferential Interaction 1,2 (Gamma_1_2)": ("Gamma_1_2", r'$\Delta\Gamma_{1,2}$', "ΔΓ₁,₂"),
                "Preferential Interaction 1,3 (Γ₁,₃)": ("Gamma_1_3", r'$\Delta\Gamma_{1,3}$', "ΔΓ₁,₃"),
            }
            
            if st.session_state.get("is_fitting_mode", False) and tern_mode in ["2D Contour Plot", "1D Slice Plot"]:
                has_eps = (st.session_state.get("fitted_eps2") is not None and 
                           st.session_state.get("fitted_eps3") is not None and 
                           st.session_state.get("fitted_eps23") is not None)
                has_epsTS = (st.session_state.get("fitted_epsTS2") is not None and 
                             st.session_state.get("fitted_epsTS3") is not None and 
                             st.session_state.get("fitted_epsTS23") is not None)
                if not has_eps:
                    properties_contour = {k: v for k, v in properties_contour.items() if "Osmotic Pressure" in k}
                elif not has_epsTS:
                    properties_contour = {k: v for k, v in properties_contour.items() if "Enthalpy" not in k and "Entropy" not in k}
            
            if tern_mode == "2D Contour Plot":
                z_name = st.selectbox("Property to plot (Contours over phi2 vs phi3)", list(properties_contour.keys()))
                z_attr, z_label, z_label_unicode = properties_contour[z_name]
                
                is_potential = any(p in z_attr for p in ["ddG", "ddH", "TddS"])
                plot_contrib = False
                if is_potential:
                    plot_contrib = st.checkbox("Plot alongside contributions (nu, chi, eps)", value=True)
                    
                if plot_contrib:
                    from plotly.subplots import make_subplots
                    
                    if "ddG" in z_attr:
                        base = "ddG"
                    elif "ddH" in z_attr:
                        base = "ddH"
                    else:
                        base = "TddS"
                        
                    suffix = "_kJ" if "_kJ" in z_attr else ""
                    
                    # Compute data
                    total_z = getattr(solved_model, f"{base}{suffix}")
                    chi_z = getattr(solved_model, f"{base}_chi{suffix}")
                    eps_z = getattr(solved_model, f"{base}_eps{suffix}")
                    if base != "ddH":
                        nu_z = getattr(solved_model, f"{base}_nu{suffix}")
                    
                    pfig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=(
                            f"Total {z_label_unicode}",
                            "ν (Excluded Volume) Contribution" if base != "ddH" else "No ν Contribution (Enthalpy)",
                            "χ (Non-ideal mixing) Contribution",
                            "ε (Soft interaction) Contribution"
                        ),
                        horizontal_spacing=0.15,
                        vertical_spacing=0.15
                    )
                    
                    # Subplot 1: Total
                    pfig.add_trace(go.Contour(
                        x=solved_model.phi2[0, :],
                        y=solved_model.phi3[:, 0],
                        z=total_z,
                        colorscale=[[0, "#f8f9fa"], [1, styles.PLOT_TOTAL_COLOR]],
                        hoverongaps=False,
                        colorbar=dict(thickness=15, len=0.4, y=0.78, x=0.43, title=dict(text=z_label_unicode, font=dict(size=10))),
                        hovertemplate="φ₂: %{x:.3f}<br>φ₃: %{y:.3f}<br>Value: %{z:.4f}<extra></extra>"
                    ), row=1, col=1)
                    
                    # Subplot 2: nu
                    if base != "ddH":
                        pfig.add_trace(go.Contour(
                            x=solved_model.phi2[0, :],
                            y=solved_model.phi3[:, 0],
                            z=nu_z,
                            colorscale=[[0, "#f2f8fd"], [1, styles.PLOT_NU_COLOR]],
                            colorbar=dict(thickness=15, len=0.4, y=0.78, x=1.02, title=dict(text="ν", font=dict(size=10))),
                            hovertemplate="φ₂: %{x:.3f}<br>φ₃: %{y:.3f}<br>Value: %{z:.4f}<extra></extra>"
                        ), row=1, col=2)
                    else:
                        pfig.add_trace(go.Contour(
                            x=solved_model.phi2[0, :],
                            y=solved_model.phi3[:, 0],
                            z=np.zeros_like(solved_model.phi2),
                            colorscale=[[0, "#f2f8fd"], [1, styles.PLOT_NU_COLOR]],
                            showscale=False,
                            hovertemplate="φ₂: %{x:.3f}<br>φ₃: %{y:.3f}<br>Value: 0.0<extra></extra>"
                        ), row=1, col=2)
                        
                    # Subplot 3: chi
                    pfig.add_trace(go.Contour(
                        x=solved_model.phi2[0, :],
                        y=solved_model.phi3[:, 0],
                        z=chi_z,
                        colorscale=[[0, "#fdf6f3"], [1, styles.PLOT_CHI_COLOR]],
                        colorbar=dict(thickness=15, len=0.4, y=0.22, x=0.43, title=dict(text="χ", font=dict(size=10))),
                        hovertemplate="φ₂: %{x:.3f}<br>φ₃: %{y:.3f}<br>Value: %{z:.4f}<extra></extra>"
                    ), row=2, col=1)
                    
                    # Subplot 4: eps
                    pfig.add_trace(go.Contour(
                        x=solved_model.phi2[0, :],
                        y=solved_model.phi3[:, 0],
                        z=eps_z,
                        colorscale=[[0, "#f5faf6"], [1, styles.PLOT_EPS_COLOR]],
                        colorbar=dict(thickness=15, len=0.4, y=0.22, x=1.02, title=dict(text="ε", font=dict(size=10))),
                        hovertemplate="φ₂: %{x:.3f}<br>φ₃: %{y:.3f}<br>Value: %{z:.4f}<extra></extra>"
                    ), row=2, col=2)
                    
                    pfig.update_layout(
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        font=dict(family="Inter, DejaVu Sans", size=11),
                        margin=dict(l=60, r=60, t=50, b=60),
                        height=700,
                    )
                    for r in [1, 2]:
                        for c in [1, 2]:
                            pfig.update_xaxes(title_text="φ₂", row=r, col=c, showgrid=False, zeroline=False, showline=True, linecolor=styles.PALETTE_DARK, linewidth=1, mirror=True)
                            pfig.update_yaxes(title_text="φ₃", row=r, col=c, showgrid=False, zeroline=False, showline=True, linecolor=styles.PALETTE_DARK, linewidth=1, mirror=True)
                    
                    try:
                        _display_and_export_plotly(pfig, "fh_crowding_ternary_contour_subplots", "tern_contour_subplots", height=700)
                    except Exception as e:
                        st.error(f"Error rendering custom 2D contour plot subplots: {e}")
                else:
                    pfig = go.Figure()
                    z_data = getattr(solved_model, z_attr)
                    pfig.add_trace(go.Contour(
                        x=solved_model.phi2[0, :],
                        y=solved_model.phi3[:, 0],
                        z=z_data,
                        colorscale=[[0, "#f8f9fa"], [1, styles.PLOT_TOTAL_COLOR]],
                        colorbar=dict(title=dict(text=z_label_unicode, font=dict(size=12)), thickness=20),
                        hovertemplate="φ₂: %{x:.3f}<br>φ₃: %{y:.3f}<br>Value: %{z:.4f}<extra></extra>"
                    ))
                    pfig.update_layout(
                        title=dict(text=z_label_unicode, font=dict(family="Inter, DejaVu Sans", size=14)),
                        xaxis_title="φ₂",
                        yaxis_title="φ₃",
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        font=dict(family="Inter, DejaVu Sans", size=12),
                        margin=dict(l=60, r=20, t=50, b=60),
                        xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor=styles.PALETTE_DARK, linewidth=1, mirror=True),
                        yaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor=styles.PALETTE_DARK, linewidth=1, mirror=True),
                    )
                    try:
                        _display_and_export_plotly(pfig, "fh_crowding_ternary_contour_plot", "tern_contour_plot")
                    except Exception as e:
                        st.error(f"Error rendering custom 2D contour plot: {e}")
                    
            elif tern_mode == "3D Surface Plot":  # 3D Surface Plot
                st.write(
                    "Interactive 3D plot: model surface over the φ₂–φ₃ grid. "
                    "When experimental data is loaded, data points are overlaid as red scatter."
                )

                # Properties available for 3D surface (thermodynamic potentials only)
                props_3d = {
                    "Free Energy (ΔΔG) [kJ/mol]":    ("ddG_kJ",  "ΔΔG⁰ [kJ/mol]",  "exp_val_G"),
                    "Enthalpy (ΔΔH) [kJ/mol]":       ("ddH_kJ",  "ΔΔH⁰ [kJ/mol]",  "exp_val_H"),
                    "Entropy (TΔΔS) [kJ/mol]":        ("TddS_kJ", "TΔΔS⁰ [kJ/mol]", "exp_val_S"),
                    "Free Energy (ΔΔG) [kT]":         ("ddG",     "ΔΔG⁰ / kBT",     "exp_val_G"),
                    "Enthalpy (ΔΔH) [kT]":            ("ddH",     "ΔΔH⁰ / kBT",     "exp_val_H"),
                    "Entropy (TΔΔS) [kT]":            ("TddS",    "TΔΔS⁰ / kBT",    "exp_val_S"),
                }
                z3d_name = st.selectbox("Property (Z-axis)", list(props_3d.keys()), key="tern_3d_prop")
                z3d_attr, z3d_label, exp_val_key = props_3d[z3d_name]
                is_kJ = "_kJ" in z3d_attr
                kJ_factor = 1.0  # model already in kJ; no conversion needed

                # Build Plotly figure
                z_surface = getattr(solved_model, z3d_attr)
                pfig3d = go.Figure()

                is_fitting = st.session_state.get("is_fitting_mode", False)
                show_surface = True
                show_edge2 = True
                show_edge3 = True
                
                if is_fitting:
                    if "Free Energy" in z3d_name:
                        show_surface = st.session_state.get("fitted_eps23") is not None
                        show_edge2 = st.session_state.get("fitted_eps2") is not None
                        show_edge3 = st.session_state.get("fitted_eps3") is not None
                    elif "Enthalpy" in z3d_name or "Entropy" in z3d_name:
                        show_surface = st.session_state.get("fitted_epsTS23") is not None
                        show_edge2 = st.session_state.get("fitted_epsTS2") is not None
                        show_edge3 = st.session_state.get("fitted_epsTS3") is not None

                # Model surface
                if show_surface:
                    pfig3d.add_trace(go.Surface(
                        x=solved_model.phi2,
                        y=solved_model.phi3,
                        z=z_surface,
                        colorscale=[[0, "#f8f9fa"], [1, styles.PLOT_TOTAL_COLOR]],
                        opacity=0.65,
                        name="Model surface",
                        colorbar=dict(title=z3d_label, thickness=14, len=0.7),
                        showscale=True,
                        hovertemplate="φ₂=%{x:.3f}<br>φ₃=%{y:.3f}<br>" + z3d_label + "=%{z:.4f}<extra>Model</extra>",
                    ))

                # Binary edge lines: cosolute-2 axis (phi3 = 0) and cosolute-3 axis (phi2 = 0)
                try:
                    phi2_axis_3d = solved_model.phi2[0, :]
                    phi3_axis_3d = solved_model.phi3[:, 0]

                    # Edge along phi3 = 0 (cosolute-2 axis)
                    if show_edge2:
                        z_edge2 = z_surface[0, :]
                        pfig3d.add_trace(go.Scatter3d(
                            x=phi2_axis_3d, y=np.zeros_like(phi2_axis_3d), z=z_edge2,
                            mode="lines", name="φ₃ = 0 (cosolute 2 axis)",
                            line=dict(color=styles.PLOT_NU_COLOR, width=5),
                            hovertemplate="φ₂=%{x:.3f}<br>" + z3d_label + "=%{z:.4f}<extra>φ₃=0</extra>",
                        ))

                    # Edge along phi2 = 0 (cosolute-3 axis)
                    if show_edge3:
                        z_edge3 = z_surface[:, 0]
                        pfig3d.add_trace(go.Scatter3d(
                            x=np.zeros_like(phi3_axis_3d), y=phi3_axis_3d, z=z_edge3,
                            mode="lines", name="φ₂ = 0 (cosolute 3 axis)",
                            line=dict(color=styles.PLOT_CHI_COLOR, width=5),
                            hovertemplate="φ₃=%{y:.3f}<br>" + z3d_label + "=%{z:.4f}<extra>φ₂=0</extra>",
                        ))
                except Exception:
                    pass  # Edge lines are optional

                # Experimental scatter
                c2_key = "exp_conc2_G" if exp_val_key == "exp_val_G" else "exp_conc2_T"
                c3_key = "exp_conc3_G" if exp_val_key == "exp_val_G" else "exp_conc3_T"
                has_exp_3d = (
                    st.session_state.get(c2_key) is not None and
                    st.session_state.get(exp_val_key) is not None
                )
                if has_exp_3d:
                    exp_phi2 = convert_exp_conc(
                        st.session_state[c2_key],
                        from_type=uploaded_conc_unit, to_type="phiC",
                        model=solved_model, is_ternary=True, cosolute_idx=2
                    )
                    exp_phi3 = convert_exp_conc(
                        st.session_state[c3_key],
                        from_type=uploaded_conc_unit, to_type="phiC",
                        model=solved_model, is_ternary=True, cosolute_idx=3,
                        exp_conc3=st.session_state[c3_key]
                    )
                    exp_z = np.array(st.session_state[exp_val_key], dtype=float)
                    # Convert kcal→kJ if model axis is kJ
                    if is_kJ and uploaded_energy_unit == "kcal/mol":
                        exp_z = exp_z  # already converted by energy_mult on load
                    # Convert to kT if needed
                    if not is_kJ:
                        # kT version: divide by kBT in kJ/mol = R*T/1000? No: model uses kBT
                        # exp_z is in kJ/mol; kBT at T K = 0.008314*T kJ/mol
                        kBT = 0.008314472 * T
                        exp_z = exp_z / kBT

                    pfig3d.add_trace(go.Scatter3d(
                        x=exp_phi2, y=exp_phi3, z=exp_z,
                        mode="markers", name="Experimental",
                        marker=dict(
                            color=styles.PLOT_EXP_COLOR, size=8,
                            symbol="circle", line=dict(color="black", width=1)
                        ),
                        hovertemplate="φ₂=%{x:.3f}<br>φ₃=%{y:.3f}<br>" + z3d_label + "=%{z:.4f}<extra>Experimental</extra>",
                    ))
                elif show_exp:
                    st.info("Upload experimental data (ΔΔG / ΔΔH / TΔΔS with concentrations) to overlay scatter points.")

                pfig3d.update_layout(
                    scene=dict(
                        xaxis_title="φ₂",
                        yaxis_title="φ₃",
                        zaxis_title=z3d_label,
                        bgcolor="rgb(250,252,255)",
                    ),
                    margin=dict(l=0, r=0, t=30, b=0),
                    font=dict(family="Inter, DejaVu Sans", size=12),
                    legend=dict(
                        x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                        bordercolor="#dde3e8", borderwidth=1
                    ),
                    height=600,
                )
                try:
                    _display_and_export_plotly(pfig3d, "fh_crowding_ternary_3d_surface", "tern_3d_surface", height=620)
                except Exception as e:
                    st.error(f"Error rendering 3D surface plot: {e}")

            else: # 1D Slice Plot
                st.write("Slices through the 2D concentration space to make 1D plots.")
                
                phi2_axis = solved_model.phi2[0, :]
                phi3_axis = solved_model.phi3[:, 0]
                
                slice_type = st.selectbox(
                    "Select Slice Path",
                    ["Constant φ₃", "Constant φ₂", "Diagonal (φ₂ = φ₃)"]
                )
                
                # Pre-calculate experimental unique values for slices
                exp_phi2_list = []
                exp_phi3_list = []
                if show_exp:
                    if st.session_state.get("exp_conc2_G") is not None and st.session_state.get("exp_conc3_G") is not None:
                        try:
                            phi2_G = convert_exp_conc(st.session_state["exp_conc2_G"], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=2)
                            phi3_G = convert_exp_conc(st.session_state["exp_conc3_G"], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=3, exp_conc3=st.session_state["exp_conc3_G"])
                            exp_phi2_list.extend(phi2_G)
                            exp_phi3_list.extend(phi3_G)
                        except Exception: pass
                    if st.session_state.get("exp_conc2_T") is not None and st.session_state.get("exp_conc3_T") is not None:
                        try:
                            phi2_T = convert_exp_conc(st.session_state["exp_conc2_T"], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=2)
                            phi3_T = convert_exp_conc(st.session_state["exp_conc3_T"], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=3, exp_conc3=st.session_state["exp_conc3_T"])
                            exp_phi2_list.extend(phi2_T)
                            exp_phi3_list.extend(phi3_T)
                        except Exception: pass
                        
                def merge_close(vals, tol=0.005):
                    if not len(vals): return []
                    vals = np.sort(vals)
                    merged = [vals[0]]
                    for v in vals[1:]:
                        if v - merged[-1] > tol:
                            merged.append(v)
                    return np.round(merged, 4)

                exp_phi2_unique = merge_close(exp_phi2_list) if exp_phi2_list else []
                exp_phi3_unique = merge_close(exp_phi3_list) if exp_phi3_list else []

                if slice_type == "Constant φ₃":
                    method = st.radio("Value selection method", ["Slider", "Manual Entry", "Experimental Data"], horizontal=True, key="meth_phi3")
                    if method == "Slider":
                        val3 = st.select_slider("Select constant φ₃ value", options=list(phi3_axis))
                    elif method == "Manual Entry":
                        val3_in = st.number_input("Enter constant φ₃ value", min_value=0.0, value=float(phi3_axis[0]), step=0.01)
                        val3 = phi3_axis[np.argmin(np.abs(phi3_axis - val3_in))]
                        st.caption(f"Snapped to nearest grid value: {val3:.4f}")
                    else:
                        if len(exp_phi3_unique) > 0:
                            val3_in = st.selectbox("Select φ₃ from experimental data", exp_phi3_unique)
                            val3 = phi3_axis[np.argmin(np.abs(phi3_axis - val3_in))]
                            st.caption(f"Snapped to nearest grid value: {val3:.4f}")
                        else:
                            st.warning("No experimental data loaded.")
                            val3 = phi3_axis[0]
                            
                    idx3 = np.argmin(np.abs(phi3_axis - val3))
                    x_data = phi2_axis
                    x_label_unicode = "φ₂"
                    slicer = lambda arr2d: arr2d[idx3, :]
                elif slice_type == "Constant φ₂":
                    method = st.radio("Value selection method", ["Slider", "Manual Entry", "Experimental Data"], horizontal=True, key="meth_phi2")
                    if method == "Slider":
                        val2 = st.select_slider("Select constant φ₂ value", options=list(phi2_axis))
                    elif method == "Manual Entry":
                        val2_in = st.number_input("Enter constant φ₂ value", min_value=0.0, value=float(phi2_axis[0]), step=0.01)
                        val2 = phi2_axis[np.argmin(np.abs(phi2_axis - val2_in))]
                        st.caption(f"Snapped to nearest grid value: {val2:.4f}")
                    else:
                        if len(exp_phi2_unique) > 0:
                            val2_in = st.selectbox("Select φ₂ from experimental data", exp_phi2_unique)
                            val2 = phi2_axis[np.argmin(np.abs(phi2_axis - val2_in))]
                            st.caption(f"Snapped to nearest grid value: {val2:.4f}")
                        else:
                            st.warning("No experimental data loaded.")
                            val2 = phi2_axis[0]
                            
                    idx2 = np.argmin(np.abs(phi2_axis - val2))
                    x_data = phi3_axis
                    x_label_unicode = "φ₃"
                    slicer = lambda arr2d: arr2d[:, idx2]
                else: # Diagonal
                    x_data = np.diag(solved_model.phi2)
                    x_label_unicode = "φ₂ = φ₃"
                    slicer = lambda arr2d: np.diag(arr2d)
                    
                y_name = st.selectbox("Y-Axis Property", list(properties_contour.keys()))
                y_attr, y_label, y_label_unicode = properties_contour[y_name]
                
                is_potential = any(p in y_attr for p in ["ddG", "ddH", "TddS"])
                plot_contrib = False
                if is_potential:
                    plot_contrib = st.checkbox("Plot alongside contributions (nu, chi, eps)", value=True, key="tern_slice_contrib")
                    
                pfig = go.Figure()
                
                if plot_contrib:
                    if "ddG" in y_attr:
                        base = "ddG"
                    elif "ddH" in y_attr:
                        base = "ddH"
                    else:
                        base = "TddS"
                    suffix = "_kJ" if "_kJ" in y_attr else ""
                    
                    total_z = getattr(solved_model, f"{base}{suffix}")
                    pfig.add_trace(go.Scatter(
                        x=x_data, y=slicer(total_z),
                        name="Total Model",
                        line=dict(color=styles.PLOT_TOTAL_COLOR, width=2.5, dash="solid"),
                        hovertemplate=x_label_unicode + ": %{x:.4f}<br>Total: %{y:.4f}<extra></extra>"
                    ))
                    
                    if base != "ddH":
                        nu_z = getattr(solved_model, f"{base}_nu{suffix}")
                        pfig.add_trace(go.Scatter(
                            x=x_data, y=slicer(nu_z),
                            name="ν (Excluded Volume)",
                            line=dict(color=styles.PLOT_NU_COLOR, width=2, dash="solid"),
                            hovertemplate=x_label_unicode + ": %{x:.4f}<br>Excluded Volume: %{y:.4f}<extra></extra>"
                        ))
                        
                    chi_z = getattr(solved_model, f"{base}_chi{suffix}")
                    pfig.add_trace(go.Scatter(
                        x=x_data, y=slicer(chi_z),
                        name="χ (Non-ideal mixing)",
                        line=dict(color=styles.PLOT_CHI_COLOR, width=2, dash="solid"),
                        hovertemplate=x_label_unicode + ": %{x:.4f}<br>Mixing: %{y:.4f}<extra></extra>"
                    ))
                    
                    eps_z = getattr(solved_model, f"{base}_eps{suffix}")
                    pfig.add_trace(go.Scatter(
                        x=x_data, y=slicer(eps_z),
                        name="ε (Soft interaction)",
                        line=dict(color=styles.PLOT_EPS_COLOR, width=2, dash="solid"),
                        hovertemplate=x_label_unicode + ": %{x:.4f}<br>Soft: %{y:.4f}<extra></extra>"
                    ))
                else:
                    y_data = getattr(solved_model, y_attr)
                    pfig.add_trace(go.Scatter(
                        x=x_data, y=slicer(y_data),
                        name="Model results",
                        line=dict(color=styles.PLOT_TOTAL_COLOR, width=2.5, dash="solid"),
                        hovertemplate=x_label_unicode + ": %{x:.4f}<br>Value: %{y:.4f}<extra></extra>"
                    ))
                    
                # Overlay experimental data on 1D Slice if requested
                if show_exp:
                    if "ddG" in y_attr:
                        exp_val = st.session_state.get("exp_val_G")
                        c2_key = "exp_conc2_G"
                        c3_key = "exp_conc3_G"
                    elif "ddH" in y_attr:
                        exp_val = st.session_state.get("exp_val_H")
                        c2_key = "exp_conc2_T"
                        c3_key = "exp_conc3_T"
                    elif "TddS" in y_attr:
                        exp_val = st.session_state.get("exp_val_S")
                        c2_key = "exp_conc2_T"
                        c3_key = "exp_conc3_T"
                    else:
                        exp_val = None
                        c2_key = None
                        c3_key = None
                        
                    if exp_val is not None and st.session_state.get(c2_key) is not None:
                        exp_val = np.array(exp_val, dtype=float)
                        if "kcal" in y_attr:
                            exp_val = exp_val / 4.184
                        elif "kj" not in y_attr.lower():
                            kBT = 0.008314472 * T
                            exp_val = exp_val / kBT
                        # Convert exp concentrations to volume fractions
                        exp_x_phi = convert_exp_conc(st.session_state[c2_key], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=2)
                        exp_y_phi = convert_exp_conc(st.session_state[c3_key], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=3, exp_conc3=st.session_state[c3_key])
                        
                        # Find points in experimental dataset that align with the slice path (within tolerance)
                        tolerance = 0.005
                        if slice_type == "Constant φ₃":
                            mask = np.abs(exp_y_phi - val3) < tolerance
                            slice_exp_x = exp_x_phi[mask]
                            slice_exp_y = exp_val[mask]
                        elif slice_type == "Constant φ₂":
                            mask = np.abs(exp_x_phi - val2) < tolerance
                            slice_exp_x = exp_y_phi[mask]  # plot against phi3 axis
                            slice_exp_y = exp_val[mask]
                        else: # Diagonal
                            mask = np.abs(exp_x_phi - exp_y_phi) < tolerance
                            slice_exp_x = exp_x_phi[mask]
                            slice_exp_y = exp_val[mask]
                            
                        if len(slice_exp_x) > 0:
                            pfig.add_trace(go.Scatter(
                                x=slice_exp_x, y=slice_exp_y,
                                mode="markers",
                                name="Experimental",
                                marker=dict(
                                    color=styles.PLOT_EXP_COLOR, size=10,
                                    symbol="circle", line=dict(color="black", width=1.2)
                                ),
                                hovertemplate="Experimental<br>" + x_label_unicode + ": %{x:.4f}<br>Value: %{y:.4f}<extra></extra>"
                            ))
                            
                pfig.update_layout(
                    xaxis_title=x_label_unicode,
                    yaxis_title=y_label_unicode,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    legend=dict(
                        orientation="v", x=1.02, xanchor="left",
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#dde3e8", borderwidth=1
                    ),
                    margin=dict(l=60, r=20, t=30, b=60),
                    font=dict(family="Inter, DejaVu Sans", size=12),
                    xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor=styles.PALETTE_DARK, linewidth=1, mirror=True),
                    yaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor=styles.PALETTE_DARK, linewidth=1, mirror=True),
                )
                try:
                    _display_and_export_plotly(pfig, "fh_crowding_ternary_slice_plot", "tern_slice_plot")
                except Exception as e:
                    st.error(f"Error rendering custom 1D slice plot: {e}")

citation.render_about_and_citation()
