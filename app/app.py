import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import fh_crowding

st.set_page_config(page_title="FH Crowding Model", layout="wide")

# ---------------------------------------------------------------------------
# Cosolute database
# Parameters from: Rösgen & Auton, JACS 2023 (DOI: 10.1021/jacs.3c08702), SI.
# nu: excluded volume parameter
# chi: Flory-Huggins non-ideal mixing parameter
# chiTS: entropy component of chi
# ---------------------------------------------------------------------------
COSOLUTE_DB = {
    "Custom":     {"nu":  1.00, "chi":  0.000, "chiTS":  0.000},
    "Glycerol":   {"nu":  3.95, "chi":  0.233, "chiTS": -0.480},
    "Glucose":    {"nu":  6.27, "chi":  0.317, "chiTS": -0.317},
    "Galactose":  {"nu":  6.26, "chi":  0.350, "chiTS": -1.070},
    "Sorbitol":   {"nu":  6.71, "chi":  0.381, "chiTS": -0.290},
    "Trehalose":  {"nu": 11.70, "chi":  0.433, "chiTS": -1.120},
    "Sucrose":    {"nu": 11.90, "chi":  0.452, "chiTS": -0.854},
}

COSOLUTE_NAMES = list(COSOLUTE_DB.keys())


def _sync_cosolute(select_key: str, nu_key: str, chi_key: str, chiTS_key: str) -> None:
    """on_change callback: copy preset params into session_state for the number_input widgets."""
    params = COSOLUTE_DB[st.session_state[select_key]]
    st.session_state[nu_key]    = params["nu"]
    st.session_state[chi_key]   = params["chi"]
    st.session_state[chiTS_key] = params["chiTS"]


# ---------------------------------------------------------------------------
# Initialise session state with sensible defaults (only on first load)
# ---------------------------------------------------------------------------
_defaults = {
    # binary
    "bin_nu":   1.0,  "bin_chi":  0.1,  "bin_chiTS": -0.05,
    # ternary cosolute 2
    "nu2":      1.0,  "chi12":    0.1,  "chiTS12":  -0.05,
    # ternary cosolute 3
    "nu3":      1.0,  "chi13":    0.1,  "chiTS13":  -0.05,
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
# Sidebar — Binary model parameters
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
        help="Choosing a preset cosolute fills nu, chi, chiTS automatically. "
             "You can still edit the values below manually.",
    )

    nu    = st.sidebar.number_input("nu",    key="bin_nu",   step=0.01, format="%.4f")
    chi   = st.sidebar.number_input("chi",   key="bin_chi",  step=0.01, format="%.4f")
    chiTS = st.sidebar.number_input("chiTS", key="bin_chiTS",step=0.01, format="%.4f")

    st.sidebar.subheader("Soft Interaction Parameters")
    eps   = st.sidebar.number_input("eps",   value=0.0, step=0.01, format="%.4f")
    epsTS = st.sidebar.number_input("epsTS", value=0.0, step=0.01, format="%.4f")

    cosolute = fh_crowding.Cosolute(nu=nu, chi=chi, chiTS=chiTS)
    model = fh_crowding.BinaryCrowdingModel(
        protein=protein, cosolute=cosolute, eps=eps, epsTS=epsTS, T=T
    )

# ---------------------------------------------------------------------------
# Sidebar — Ternary model parameters
# ---------------------------------------------------------------------------
else:
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
        help="Choosing a preset fills nu2, chi12, chiTS12 automatically.",
    )
    nu2    = st.sidebar.number_input("nu2",    key="nu2",    step=0.01, format="%.4f")
    chi12  = st.sidebar.number_input("chi12",  key="chi12",  step=0.01, format="%.4f")
    chiTS12= st.sidebar.number_input("chiTS12",key="chiTS12",step=0.01, format="%.4f")
    eps2   = st.sidebar.number_input("eps2",   value=0.0, step=0.01, format="%.4f")
    epsTS2 = st.sidebar.number_input("epsTS2", value=0.0, step=0.01, format="%.4f")

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
        help="Choosing a preset fills nu3, chi13, chiTS13 automatically.",
    )
    nu3    = st.sidebar.number_input("nu3",    key="nu3",    step=0.01, format="%.4f")
    chi13  = st.sidebar.number_input("chi13",  key="chi13",  step=0.01, format="%.4f")
    chiTS13= st.sidebar.number_input("chiTS13",key="chiTS13",step=0.01, format="%.4f")
    eps3   = st.sidebar.number_input("eps3",   value=0.0, step=0.01, format="%.4f")
    epsTS3 = st.sidebar.number_input("epsTS3", value=0.0, step=0.01, format="%.4f")

    # --- Cosolute–cosolute interactions ---
    st.sidebar.subheader("Cosolute–Cosolute Interactions")
    chi23   = st.sidebar.number_input("chi23",   value=0.0, step=0.01, format="%.4f")
    chiTS23 = st.sidebar.number_input("chiTS23", value=0.0, step=0.01, format="%.4f")
    eps23   = st.sidebar.number_input("eps23",   value=0.0, step=0.01, format="%.4f")
    epsTS23 = st.sidebar.number_input("epsTS23", value=0.0, step=0.01, format="%.4f")

    cosolutes = fh_crowding.CosoluteMixture(
        nu2=nu2, nu3=nu3,
        chi12=chi12, chi13=chi13, chi23=chi23,
        chiTS12=chiTS12, chiTS13=chiTS13, chiTS23=chiTS23,
    )
    model = fh_crowding.TernaryCrowdingModel(
        protein=protein, cosolutes=cosolutes,
        eps2=eps2, eps3=eps3, eps23=eps23,
        epsTS2=epsTS2, epsTS3=epsTS3, epsTS23=epsTS23,
        T=T,
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["Simulation", "Data Fitting"])

with tab1:
    st.header("Forward Simulation")
    if st.button("Run Simulation"):
        with st.spinner("Solving equilibrium..."):
            try:
                model.solve_equil()
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
