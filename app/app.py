import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import fh_crowding

st.set_page_config(page_title="FH Crowding Model", layout="wide")

st.title("FH Crowding Thermodynamic Model")

# Sidebar - Parameters
st.sidebar.header("Model Configuration")

model_type = st.sidebar.selectbox("Model Type", ["Binary Crowding Model", "Ternary Crowding Model"])

st.sidebar.subheader("Protein")
SASA = st.sidebar.number_input("SASA", value=419.0)
protein = fh_crowding.Protein(SASA=SASA)

st.sidebar.subheader("Temperature")
use_T = st.sidebar.checkbox("Include Temperature", value=True)
if use_T:
    T = st.sidebar.number_input("Temperature (K)", value=298.15)
else:
    T = 298.15

if model_type == "Binary Crowding Model":
    st.sidebar.subheader("Cosolute Parameters")
    nu = st.sidebar.number_input("nu", value=1.0)
    chi = st.sidebar.number_input("chi", value=0.1)
    chiTS = st.sidebar.number_input("chiTS", value=-0.05)
    eps = st.sidebar.number_input("eps", value=0.0)
    epsTS = st.sidebar.number_input("epsTS", value=0.0)
    
    cosolute = fh_crowding.Cosolute(nu=nu, chi=chi, chiTS=chiTS)
    model = fh_crowding.BinaryCrowdingModel(protein=protein, cosolute=cosolute, eps=eps, epsTS=epsTS, T=T)

else:
    st.sidebar.subheader("Cosolute 2 Parameters")
    nu2 = st.sidebar.number_input("nu2", value=1.0)
    eps2 = st.sidebar.number_input("eps2", value=0.0)
    epsTS2 = st.sidebar.number_input("epsTS2", value=0.0)
    
    st.sidebar.subheader("Cosolute 3 Parameters")
    nu3 = st.sidebar.number_input("nu3", value=1.0)
    eps3 = st.sidebar.number_input("eps3", value=0.0)
    epsTS3 = st.sidebar.number_input("epsTS3", value=0.0)
    
    st.sidebar.subheader("Interactions (chi)")
    chi12 = st.sidebar.number_input("chi12", value=0.1)
    chiTS12 = st.sidebar.number_input("chiTS12", value=-0.05)
    chi13 = st.sidebar.number_input("chi13", value=0.1)
    chiTS13 = st.sidebar.number_input("chiTS13", value=-0.05)
    chi23 = st.sidebar.number_input("chi23", value=0.0)
    chiTS23 = st.sidebar.number_input("chiTS23", value=0.0)
    
    st.sidebar.subheader("Interactions (eps)")
    eps23 = st.sidebar.number_input("eps23", value=0.0)
    epsTS23 = st.sidebar.number_input("epsTS23", value=0.0)
    
    cosolutes = fh_crowding.CosoluteMixture(nu2=nu2, nu3=nu3, chi12=chi12, chi13=chi13, chi23=chi23,
                                            chiTS12=chiTS12, chiTS13=chiTS13, chiTS23=chiTS23)
    model = fh_crowding.TernaryCrowdingModel(protein=protein, cosolutes=cosolutes, eps2=eps2, eps3=eps3, 
                                             eps23=eps23, epsTS2=epsTS2, epsTS3=epsTS3, epsTS23=epsTS23, T=T)

tab1, tab2 = st.tabs(["Simulation", "Data Fitting"])

with tab1:
    st.header("Forward Simulation")
    if st.button("Run Simulation"):
        with st.spinner("Solving equilibrium..."):
            model.solve_equil()
            model.to_pandas()
            st.success("Simulation complete!")
            
            st.subheader("Data Export")
            csv = model.results.to_csv(index=False)
            st.download_button("Download Results CSV", data=csv, file_name="simulation_results.csv", mime="text/csv")
            
            st.subheader("Plots")
            if model_type == "Binary Crowding Model":
                plotter = fh_crowding.BinaryPlotter(model)
                st.pyplot(plotter.plot_results().figure)
            else:
                plotter = fh_crowding.TernaryPlotter(model)
                # Ternary models plot inline, so we have to capture the figures or rewrite the plotter to return figures
                st.info("Ternary plotting currently relies on pyplot.show(), so it might plot directly to the backend. We'll show a sample plot.")
                # Since the plotter calls plt.show(), we override it locally or use it correctly if modified.
                # In this basic version we just trigger one plot as example
                import matplotlib.pyplot as plt
                plotter.plot_ddG()
                st.pyplot(plt.gcf())

with tab2:
    st.header("Data Fitting")
    st.write("Upload experimental data to fit the soft interaction parameters (`eps`, `epsTS`).")
    
    conc_type = st.selectbox("Concentration Type", ["phi", "molar", "molal"])
    
    if model_type == "Binary Crowding Model":
        bin_format = st.radio("CSV Format", ["Format 1: One file (conc, dG, dH, TdS)", "Format 2: Three files (dG, dH, TdS)"])
        
        if "Format 1" in bin_format:
            file = st.file_uploader("Upload CSV")
            if file and st.button("Fit epsTS"):
                df = pd.read_csv(file)
                if 'dG' in df.columns:
                    model.fit_eps(df['concentration'].values, df['dG'].values, concentration_type=conc_type)
                    st.write("Fitted eps:", model.eps)
                if 'dH' in df.columns and 'TdS' in df.columns:
                    model.fit_epsTS(df['concentration'].values, df['dH'].values, df['TdS'].values, concentration_type=conc_type)
                    st.write("Fitted epsTS:", model.epsTS)
        else:
            f1 = st.file_uploader("Upload dG CSV")
            f2 = st.file_uploader("Upload dH CSV")
            f3 = st.file_uploader("Upload TdS CSV")
            if f2 and f3 and st.button("Fit epsTS"):
                df_H = pd.read_csv(f2)
                df_S = pd.read_csv(f3)
                model.fit_epsTS(df_H['concentration'].values, df_H['dH'].values, df_S['TdS'].values, concentration_type=conc_type)
                st.write("Fitted epsTS:", model.epsTS)
    else:
        tern_format = st.radio("CSV Format", ["Format 1: Columns (conc2, conc3, potential)", "Format 2: 2D Matrices"])
        f1 = st.file_uploader("Upload dG CSV (Ternary)")
        f2 = st.file_uploader("Upload dH CSV (Ternary)")
        f3 = st.file_uploader("Upload TdS CSV (Ternary)")
        
        if f1 and st.button("Fit eps2 & eps3 (from dG)"):
            if "Format 1" in tern_format:
                df = pd.read_csv(f1)
                model.fit_eps(df['conc2'].values, df['conc3'].values, df['dG'].values, concentration_type=conc_type)
                st.write("Fitted eps2:", model.eps2, "eps3:", model.eps3)
            else:
                st.warning("Matrix parsing not yet implemented dynamically. Ensure standard formatting.")

