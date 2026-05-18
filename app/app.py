import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
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
    eps   = st.sidebar.number_input("eps",   value=0.0, step=0.01, format="%.4f")
    epsTS = st.sidebar.number_input("epsTS", value=0.0, step=0.01, format="%.4f")

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
    eps2    = st.sidebar.number_input("eps2",    value=0.0, step=0.01, format="%.4f")
    epsTS2  = st.sidebar.number_input("epsTS2",  value=0.0, step=0.01, format="%.4f")

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
    eps3    = st.sidebar.number_input("eps3",    value=0.0, step=0.01, format="%.4f")
    epsTS3  = st.sidebar.number_input("epsTS3",  value=0.0, step=0.01, format="%.4f")

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
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["Simulation", "Data Fitting"])

with tab1:
    st.header("Forward Simulation")
    if st.button("Run Simulation"):
        progress_bar = st.progress(0, text="Solving equilibrium... 0%")

        def update_progress(frac: float) -> None:
            progress_bar.progress(frac, text=f"Solving equilibrium... {int(frac * 100)}%")

        try:
            kw = {"callback": update_progress}
            if model_type == "Ternary Crowding Model":
                kw["print_msg"] = False  # suppress per-point fsolve warnings in the app
            model.solve_equil(**kw)
            progress_bar.progress(1.0, text="Equilibrium solved!")

            model.to_pandas()
            st.success("Simulation complete!")

            st.subheader("Data Export")
            csv = model.results.to_csv(index=False)
            st.download_button(
                "Download Results CSV",
                data=csv,
                file_name="simulation_results.csv",
                mime="text/csv",
            )

            st.subheader("Plots")
            if model_type == "Binary Crowding Model":
                plotter = fh_crowding.BinaryPlotter(model)
                fig = plotter.plot_results()
                st.pyplot(fig)
            else:
                plotter = fh_crowding.TernaryPlotter(model)
                fig = plotter.plot_ddG()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error during simulation: {e}")

with tab2:
    st.header("Data Fitting")
    st.write("Upload experimental data to fit the soft interaction parameters (`eps`, `epsTS`).")

    conc_type = st.selectbox("Concentration Type", ["phi", "molar", "molal"])

    if model_type == "Binary Crowding Model":
        bin_format = st.radio(
            "CSV Format",
            ["Format 1: One file (conc, dG, dH, TdS)", "Format 2: Three files (dG, dH, TdS)"],
        )

        if "Format 1" in bin_format:
            file = st.file_uploader("Upload CSV")
            if file and st.button("Fit Data"):
                try:
                    df = pd.read_csv(file)
                    if "dG" not in df.columns and (
                        "dH" not in df.columns or "TdS" not in df.columns
                    ):
                        st.error(
                            "Uploaded CSV is missing expected columns. "
                            "Need 'dG' or both 'dH' and 'TdS'."
                        )
                    else:
                        if "dG" in df.columns:
                            model.fit_eps(
                                df["concentration"].values,
                                df["dG"].values,
                                concentration_type=conc_type,
                            )
                            st.write("Fitted eps:", model.eps)
                        if "dH" in df.columns and "TdS" in df.columns:
                            model.fit_epsTS(
                                df["concentration"].values,
                                df["dH"].values,
                                df["TdS"].values,
                                concentration_type=conc_type,
                            )
                            st.write("Fitted epsTS:", model.epsTS)
                except Exception as e:
                    st.error(f"Error during fitting: {e}")
        else:
            f1 = st.file_uploader("Upload dG CSV")
            f2 = st.file_uploader("Upload dH CSV")
            f3 = st.file_uploader("Upload TdS CSV")

            if f1 and st.button("Fit eps (from dG)"):
                try:
                    df_G = pd.read_csv(f1)
                    if "concentration" not in df_G.columns or "dG" not in df_G.columns:
                        st.error("CSV must contain 'concentration' and 'dG' columns.")
                    else:
                        model.fit_eps(
                            df_G["concentration"].values,
                            df_G["dG"].values,
                            concentration_type=conc_type,
                        )
                        st.write("Fitted eps:", model.eps)
                except Exception as e:
                    st.error(f"Error during fitting: {e}")

            if f2 and f3 and st.button("Fit epsTS (from dH, TdS)"):
                try:
                    df_H = pd.read_csv(f2)
                    df_S = pd.read_csv(f3)
                    if (
                        "concentration" not in df_H.columns
                        or "dH" not in df_H.columns
                        or "TdS" not in df_S.columns
                    ):
                        st.error(
                            "CSVs must contain 'concentration', 'dH', and 'TdS' columns."
                        )
                    else:
                        model.fit_epsTS(
                            df_H["concentration"].values,
                            df_H["dH"].values,
                            df_S["TdS"].values,
                            concentration_type=conc_type,
                        )
                        st.write("Fitted epsTS:", model.epsTS)
                except Exception as e:
                    st.error(f"Error during fitting: {e}")

    else:
        tern_format = st.radio(
            "CSV Format",
            ["Format 1: Columns (conc2, conc3, potential)", "Format 2: 2D Matrices"],
        )
        f1 = st.file_uploader("Upload dG CSV (Ternary)")
        f2 = st.file_uploader("Upload dH CSV (Ternary)")
        f3 = st.file_uploader("Upload TdS CSV (Ternary)")

        if f1 and st.button("Fit eps2 & eps3 (from dG)"):
            if "Format 1" in tern_format:
                try:
                    df = pd.read_csv(f1)
                    if (
                        "conc2" not in df.columns
                        or "conc3" not in df.columns
                        or "dG" not in df.columns
                    ):
                        st.error("CSV must contain 'conc2', 'conc3', and 'dG' columns.")
                    else:
                        model.fit_eps(
                            df["conc2"].values,
                            df["conc3"].values,
                            df["dG"].values,
                            concentration_type=conc_type,
                        )
                        st.write("Fitted eps2:", model.eps2, "eps3:", model.eps3)
                except Exception as e:
                    st.error(f"Error during fitting: {e}")
            else:
                st.warning(
                    "Matrix parsing not yet implemented dynamically. "
                    "Ensure standard formatting."
                )

        if f2 and f3 and st.button("Fit epsTS2 & epsTS3 (from dH, TdS)"):
            if "Format 1" in tern_format:
                try:
                    df_H = pd.read_csv(f2)
                    df_S = pd.read_csv(f3)
                    if (
                        "conc2" not in df_H.columns
                        or "conc3" not in df_H.columns
                        or "dH" not in df_H.columns
                        or "TdS" not in df_S.columns
                    ):
                        st.error(
                            "CSVs must contain 'conc2', 'conc3', 'dH', and 'TdS' columns."
                        )
                    else:
                        model.fit_epsTS(
                            df_H["conc2"].values,
                            df_H["conc3"].values,
                            df_H["dH"].values,
                            df_S["TdS"].values,
                            concentration_type=conc_type,
                        )
                        st.write(
                            "Fitted epsTS2:", model.epsTS2, "epsTS3:", model.epsTS3
                        )
                except Exception as e:
                    st.error(f"Error during fitting: {e}")
            else:
                st.warning(
                    "Matrix parsing not yet implemented dynamically. "
                    "Ensure standard formatting."
                )
