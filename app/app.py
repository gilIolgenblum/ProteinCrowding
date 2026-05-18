import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import fh_crowding

st.set_page_config(page_title="FH Crowding Model", layout="wide")

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
    params = COSOLUTE_DB[st.session_state[select_key]]
    st.session_state[nu_key]    = params["nu"]
    st.session_state[chi_key]   = params["chi"]
    st.session_state[chiTS_key] = params["chiTS"]


def _sync_cosolute_pair() -> None:
    """Fill all ternary parameters from a cosolute-pair preset."""
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
# Initialise session state defaults (only on first load)
# ---------------------------------------------------------------------------
_defaults = {
    # binary cosolute
    "bin_nu":    1.0,  "bin_chi":   0.1,  "bin_chiTS":  -0.05,
    # ternary cosolute 2
    "nu2":       1.0,  "chi12":     0.1,  "chiTS12":    -0.05,
    # ternary cosolute 3
    "nu3":       1.0,  "chi13":     0.1,  "chiTS13":    -0.05,
    # ternary cross-interaction
    "chi23":     0.0,  "chiTS23":   0.0,
    
    # experimental data keys
    "exp_conc_G": None, "exp_ddG": None, "err_ddG": None,
    "exp_conc_T": None, "exp_ddH": None, "exp_TddS": None, "err_ddH": None, "err_TddS": None,
    "exp_conc2": None, "exp_conc3": None, "exp_val_G": None, "exp_val_H": None, "exp_val_S": None,
    "exp_data_loaded": False,
    
    # fitted parameters tracker (to persist fitted parameters to sidebar input values)
    "fitted_eps": None, "fitted_epsTS": None,
    "fitted_eps2": None, "fitted_eps3": None, "fitted_epsTS2": None, "fitted_epsTS3": None
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# Sidebar — common
# ---------------------------------------------------------------------------
st.title("FH Crowding Thermodynamic Model")

st.sidebar.header("Model Configuration")
model_type = st.sidebar.selectbox(
    "Model Type", ["Binary Crowding Model", "Ternary Crowding Model"]
)

st.sidebar.subheader("Protein")
SASA = st.sidebar.number_input("SASA", value=419.0)
protein = fh_crowding.Protein(SASA=SASA)

st.sidebar.subheader("Temperature")
use_T = st.sidebar.checkbox("Include Temperature", value=True)
T = st.sidebar.number_input("Temperature (K)", value=298.15) if use_T else 298.15


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
    
    # We display fitted parameter if available, otherwise manual input
    eps_val_default = st.session_state["fitted_eps"] if st.session_state["fitted_eps"] is not None else 0.0
    eps = st.sidebar.number_input("eps", value=eps_val_default, step=0.01, format="%.4f", key="bin_eps_input")
    
    epsTS_val_default = st.session_state["fitted_epsTS"] if st.session_state["fitted_epsTS"] is not None else 0.0
    epsTS = st.sidebar.number_input("epsTS", value=epsTS_val_default, step=0.01, format="%.4f", key="bin_epsts_input")

    st.sidebar.subheader("Concentration Grid")
    dphiC = st.sidebar.number_input(
        "dphiC (step size)",
        value=0.001,
        min_value=1e-5,
        max_value=0.05,
        step=0.0005,
        format="%.5f",
        help="Concentration grid step size. Smaller = finer grid, slower simulation. "
             "Package default: 0.0001. App default: 0.001 (10× faster).",
    )

    cosolute = fh_crowding.Cosolute(nu=nu, chi=chi, chiTS=chiTS)
    model = fh_crowding.BinaryCrowdingModel(
        protein=protein, cosolute=cosolute, eps=eps, epsTS=epsTS,
        dphiC=dphiC, T=T,
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
    
    eps2_val_default = st.session_state["fitted_eps2"] if st.session_state["fitted_eps2"] is not None else 0.0
    eps2    = st.sidebar.number_input("eps2",    value=eps2_val_default, step=0.01, format="%.4f", key="tern_eps2_input")
    
    epsTS2_val_default = st.session_state["fitted_epsTS2"] if st.session_state["fitted_epsTS2"] is not None else 0.0
    epsTS2  = st.sidebar.number_input("epsTS2",  value=epsTS2_val_default, step=0.01, format="%.4f", key="tern_epsts2_input")

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
    
    eps3_val_default = st.session_state["fitted_eps3"] if st.session_state["fitted_eps3"] is not None else 0.0
    eps3    = st.sidebar.number_input("eps3",    value=eps3_val_default, step=0.01, format="%.4f", key="tern_eps3_input")
    
    epsTS3_val_default = st.session_state["fitted_epsTS3"] if st.session_state["fitted_epsTS3"] is not None else 0.0
    epsTS3  = st.sidebar.number_input("epsTS3",  value=epsTS3_val_default, step=0.01, format="%.4f", key="tern_epsts3_input")

    # --- Cosolute–cosolute interactions ---
    st.sidebar.subheader("Cosolute–Cosolute Interactions")
    chi23   = st.sidebar.number_input("chi23",   key="chi23",   step=0.01, format="%.4f")
    chiTS23 = st.sidebar.number_input("chiTS23", key="chiTS23", step=0.01, format="%.4f")
    eps23   = st.sidebar.number_input("eps23",   value=0.0, step=0.01, format="%.4f")
    epsTS23 = st.sidebar.number_input("epsTS23", value=0.0, step=0.01, format="%.4f")

    # --- Concentration grid ---
    st.sidebar.subheader("Concentration Grid")
    dphi2 = st.sidebar.number_input(
        "dphi2 (step size for cosolute 2)",
        value=0.001,
        min_value=1e-5,
        max_value=0.05,
        step=0.0005,
        format="%.5f",
        help="Grid step for cosolute 2 axis. Package default: 0.0001. App default: 0.001.",
    )
    dphi3 = st.sidebar.number_input(
        "dphi3 (step size for cosolute 3)",
        value=0.001,
        min_value=1e-5,
        max_value=0.05,
        step=0.0005,
        format="%.5f",
        help="Grid step for cosolute 3 axis. Package default: 0.0001. App default: 0.001.",
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
        T=T,
    )


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
        
    with col_u2:
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
                    df = pd.read_csv(f)
                    st.dataframe(df.head(), use_container_width=True)
                    if "concentration" in df.columns:
                        st.session_state["exp_conc_G"] = df["concentration"].values
                        st.session_state["exp_conc_T"] = df["concentration"].values
                        st.session_state["exp_data_loaded"] = True
                        
                        if "dG" in df.columns:
                            st.session_state["exp_ddG"] = df["dG"].values
                        if "err_dG" in df.columns:
                            st.session_state["err_ddG"] = df["err_dG"].values
                        else:
                            st.session_state["err_ddG"] = np.nan
                            
                        if "dH" in df.columns:
                            st.session_state["exp_ddH"] = df["dH"].values
                        if "err_dH" in df.columns:
                            st.session_state["err_ddH"] = df["err_dH"].values
                        else:
                            st.session_state["err_ddH"] = np.nan
                            
                        if "TdS" in df.columns:
                            st.session_state["exp_TddS"] = df["TdS"].values
                        if "err_TdS" in df.columns:
                            st.session_state["err_TddS"] = df["err_TdS"].values
                        else:
                            st.session_state["err_TddS"] = np.nan
                            
                        st.success("Experimental dataset loaded successfully!")
            else:
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    f_G = st.file_uploader("Upload dG CSV (concentration, dG)", type=["csv"], key="bin_g_uploader")
                with col_f2:
                    f_H = st.file_uploader("Upload dH CSV (concentration, dH)", type=["csv"], key="bin_h_uploader")
                with col_f3:
                    f_S = st.file_uploader("Upload TdS CSV (concentration, TdS)", type=["csv"], key="bin_s_uploader")
                
                if f_G:
                    df = pd.read_csv(f_G)
                    if "concentration" in df.columns and "dG" in df.columns:
                        st.session_state["exp_conc_G"] = df["concentration"].values
                        st.session_state["exp_ddG"] = df["dG"].values
                        st.session_state["err_ddG"] = df["err_dG"].values if "err_dG" in df.columns else np.nan
                        st.session_state["exp_data_loaded"] = True
                        st.success("dG experimental data loaded!")
                        
                if f_H:
                    df = pd.read_csv(f_H)
                    if "concentration" in df.columns and "dH" in df.columns:
                        st.session_state["exp_conc_T"] = df["concentration"].values
                        st.session_state["exp_ddH"] = df["dH"].values
                        st.session_state["err_ddH"] = df["err_dH"].values if "err_dH" in df.columns else np.nan
                        st.session_state["exp_data_loaded"] = True
                        st.success("dH experimental data loaded!")
                        
                if f_S:
                    df = pd.read_csv(f_S)
                    if "concentration" in df.columns and "TdS" in df.columns:
                        st.session_state["exp_conc_T"] = df["concentration"].values
                        st.session_state["exp_TddS"] = df["TdS"].values
                        st.session_state["err_TddS"] = df["err_TdS"].values if "err_TdS" in df.columns else np.nan
                        st.session_state["exp_data_loaded"] = True
                        st.success("TdS experimental data loaded!")
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
                    df = pd.read_csv(f_G)
                    if "conc2" in df.columns and "conc3" in df.columns and "dG" in df.columns:
                        st.session_state["exp_conc2"] = df["conc2"].values
                        st.session_state["exp_conc3"] = df["conc3"].values
                        st.session_state["exp_val_G"] = df["dG"].values
                        st.session_state["exp_data_loaded"] = True
                        st.success("Ternary dG data loaded!")
                if f_H:
                    df = pd.read_csv(f_H)
                    if "conc2" in df.columns and "conc3" in df.columns and "dH" in df.columns:
                        st.session_state["exp_conc2"] = df["conc2"].values
                        st.session_state["exp_conc3"] = df["conc3"].values
                        st.session_state["exp_val_H"] = df["dH"].values
                        st.session_state["exp_data_loaded"] = True
                        st.success("Ternary dH data loaded!")
                if f_S:
                    df = pd.read_csv(f_S)
                    if "conc2" in df.columns and "conc3" in df.columns and "TdS" in df.columns:
                        st.session_state["exp_conc2"] = df["conc2"].values
                        st.session_state["exp_conc3"] = df["conc3"].values
                        st.session_state["exp_val_S"] = df["TdS"].values
                        st.session_state["exp_data_loaded"] = True
                        st.success("Ternary TdS data loaded!")
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
        csv = solved_model.results.to_csv(index=False)
        st.download_button(
            "📥 Download Simulation Results (CSV)",
            data=csv,
            file_name="simulation_results.csv",
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
        
        # Fit concentration type dropdown (must match calculations)
        fit_conc_type = st.selectbox(
            "Fitting Model concentration type",
            ["phi", "molar", "molal"] if model_type == "Binary Crowding Model" else ["phi", "molal"],
            key="fitting_model_conc_type",
            help="This matches concentration_type inside the fitting algorithms."
        )
        
        st.markdown("---")
        
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
                            # Run fit
                            model.fit_eps(
                                st.session_state["exp_conc_G"],
                                st.session_state["exp_ddG"],
                                concentration_type=fit_conc_type
                            )
                            fit_progress.progress(0.5, text="Solving equilibrium with fitted eps...")
                            # Resolve model
                            model.solve_equil()
                            model.to_pandas()
                            fit_progress.progress(1.0, text="Fit & Simulation updated!")
                            
                            # Save state
                            st.session_state["fitted_eps"] = model.eps
                            st.session_state["solved_model"] = model
                            st.session_state["solved_model_type"] = model_type
                            st.success(f"Successfully fitted eps: {model.eps:.4f}")
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
                            # Run fit
                            model.fit_epsTS(
                                st.session_state["exp_conc_T"],
                                st.session_state["exp_ddH"],
                                st.session_state["exp_TddS"],
                                concentration_type=fit_conc_type
                            )
                            fit_progress.progress(0.5, text="Solving equilibrium with fitted epsTS...")
                            # Resolve model
                            model.solve_equil()
                            model.to_pandas()
                            fit_progress.progress(1.0, text="Fit & Simulation updated!")
                            
                            # Save state
                            st.session_state["fitted_epsTS"] = model.epsTS
                            st.session_state["solved_model"] = model
                            st.session_state["solved_model_type"] = model_type
                            st.success(f"Successfully fitted epsTS: {model.epsTS:.4f}")
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
                            # Run fit
                            model.fit_eps(
                                st.session_state["exp_conc2"],
                                st.session_state["exp_conc3"],
                                st.session_state["exp_val_G"],
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
                            st.session_state["solved_model"] = model
                            st.session_state["solved_model_type"] = model_type
                            st.success(f"Successfully fitted eps2: {model.eps2:.4f}, eps3: {model.eps3:.4f}")
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
                            # Run fit
                            model.fit_epsTS(
                                st.session_state["exp_conc2"],
                                st.session_state["exp_conc3"],
                                st.session_state["exp_val_H"],
                                st.session_state["exp_val_S"],
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
                            st.session_state["solved_model"] = model
                            st.session_state["solved_model_type"] = model_type
                            st.success(f"Successfully fitted epsTS2: {model.epsTS2:.4f}, epsTS3: {model.epsTS3:.4f}")
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
        show_exp = st.checkbox("Overlay uploaded experimental data on plots", value=True)
        
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
                folding_plot = st.checkbox("Plot folding (kJ) vs unfolding (kcal)", value=True, key="bin_plot_folding")
            
            # Setup experimental values to overlay if enabled
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

            plotter = fh_crowding.BinaryPlotter(solved_model)
            fig = plotter.plot_results(**plot_kwargs)
            st.pyplot(fig)
            
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
            if show_exp and st.session_state.get("exp_conc2") is not None:
                # Convert exp points concentration to phi
                exp_x_phi = convert_exp_conc(st.session_state["exp_conc2"], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=2)
                exp_y_phi = convert_exp_conc(st.session_state["exp_conc3"], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=3, exp_conc3=st.session_state["exp_conc3"])
                
                # Iterate axes and scatter the experimental points
                for ax in fig.get_axes():
                    # only scatter on contour subplots (skip colorbar axes which have no set_xlabel or set_ylabel)
                    if hasattr(ax, 'get_xlabel') and ax.get_xlabel() == r'$\phi_2$':
                        ax.scatter(exp_x_phi, exp_y_phi, color='red', edgecolor='white', s=25, label='Experimental', zorder=10)
                        
            st.pyplot(fig)

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
            
            # Check if Y-Axis is one of the potentials for contribution plotting
            is_potential = False
            pot_type = None
            pot_unit = "kJ"
            if "ddA" in y_attr:
                is_potential = True
                pot_type = "ddA"
                pot_unit = "kcal" if "kcal" in y_attr else "kJ"
            elif "ddE" in y_attr:
                is_potential = True
                pot_type = "ddE"
                pot_unit = "kcal" if "kcal" in y_attr else "kJ"
            elif "TddS" in y_attr:
                is_potential = True
                pot_type = "TddS"
                pot_unit = "kcal" if "kcal" in y_attr else "kJ"
                
            plot_contrib = False
            if is_potential:
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
                exp_y = None
                err_y = None
                exp_x = None
                
                if "ddA" in y_attr:
                    exp_y = st.session_state.get("exp_ddG")
                    err_y = st.session_state.get("err_ddG")
                    exp_x = st.session_state.get("exp_conc_G")
                    if "kcal" in y_attr and exp_y is not None:
                        exp_y = exp_y / 4.184
                        if err_y is not None:
                            err_y = err_y / 4.184
                elif "ddE" in y_attr:
                    exp_y = st.session_state.get("exp_ddH")
                    err_y = st.session_state.get("err_ddH")
                    exp_x = st.session_state.get("exp_conc_T")
                    if "kcal" in y_attr and exp_y is not None:
                        exp_y = exp_y / 4.184
                        if err_y is not None:
                            err_y = err_y / 4.184
                elif "TddS" in y_attr:
                    exp_y = st.session_state.get("exp_TddS")
                    err_y = st.session_state.get("err_TddS")
                    exp_x = st.session_state.get("exp_conc_T")
                    if "kcal" in y_attr and exp_y is not None:
                        exp_y = exp_y / 4.184
                        if err_y is not None:
                            err_y = err_y / 4.184
                            
                if exp_y is not None and exp_x is not None:
                    # Match experimental concentration type to plotted axis unit
                    exp_x_converted = convert_exp_conc(
                        exp_x,
                        from_type=uploaded_conc_unit,
                        to_type=x_attr if x_attr != "phi" else "phiC",
                        model=solved_model
                    )
                    # Scatter experimental data points
                    if err_y is not None and not np.all(np.isnan(err_y)):
                        ax.errorbar(
                            exp_x_converted, exp_y, yerr=err_y,
                            fmt='o', color='red', ecolor='red', capsize=4,
                            label='Experimental', zorder=5
                        )
                    else:
                        ax.scatter(
                            exp_x_converted, exp_y,
                            color='red', edgecolor='black', s=45,
                            label='Experimental', zorder=5
                        )
                    ax.legend()
                    
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, linestyle=":", alpha=0.6)
            st.pyplot(fig)
            
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
                    st.pyplot(fig)
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
                    if show_exp and st.session_state.get("exp_conc2") is not None:
                        exp_x_phi = convert_exp_conc(st.session_state["exp_conc2"], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=2)
                        exp_y_phi = convert_exp_conc(st.session_state["exp_conc3"], from_type=uploaded_conc_unit, to_type="phiC", model=solved_model, is_ternary=True, cosolute_idx=3, exp_conc3=st.session_state["exp_conc3"])
                        ax.scatter(exp_x_phi, exp_y_phi, color='red', edgecolor='white', s=45, label='Experimental', zorder=10)
                        ax.legend()
                        
                    st.pyplot(fig)
                    
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
                st.pyplot(fig)
