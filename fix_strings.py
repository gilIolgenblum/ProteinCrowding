import re

with open("app/app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Sidebar variables
content = content.replace('"εₜₛ (entropy component of ε)"', '"εTS (entropy component of ε)"')
content = content.replace('"εₜₛ₂ (entropy component of ε)"', '"εTS₂ (entropy component of ε)"')
content = content.replace('"εₜₛ₃ (entropy component of ε)"', '"εTS₃ (entropy component of ε)"')
content = content.replace('"εₜₛ₂₃ (entropy component)"', '"εTS₂₃ (entropy component)"')

# Fitting buttons
content = content.replace('"Fit εₜₛ (from ΔΔH, TΔΔS)"', '"Fit εTS (from ΔΔH, TΔΔS)"')
content = content.replace('"Fit ε₂ (phi3 = 0)"', '"Fit ε₂ (φ₃ = 0)"')
content = content.replace('"Fit ε₃ (phi2 = 0)"', '"Fit ε₃ (φ₂ = 0)"')
content = content.replace('"Fit εₜₛ₂ (phi3 = 0)"', '"Fit εTS₂ (φ₃ = 0)"')
content = content.replace('"Fit εₜₛ₃ (phi2 = 0)"', '"Fit εTS₃ (φ₂ = 0)"')
content = content.replace('"Fit εₜₛ₂₃ (All data)"', '"Fit εTS₂₃ (All data)"')

# Progress texts
content = content.replace('text="Fitting eps..."', 'text="Fitting ε..."')
content = content.replace('text="Fitting epsTS..."', 'text="Fitting εTS..."')
content = content.replace('text="Fitting eps2 (where phi3 = 0)..."', 'text="Fitting ε₂ (where φ₃ = 0)..."')
content = content.replace('text="Fitting eps3 (where phi2 = 0)..."', 'text="Fitting ε₃ (where φ₂ = 0)..."')
content = content.replace('text="Fitting eps23 (using all data)..."', 'text="Fitting ε₂₃ (using all data)..."')
content = content.replace('text="Fitting epsTS2 (where phi3 = 0)..."', 'text="Fitting εTS₂ (where φ₃ = 0)..."')
content = content.replace('text="Fitting epsTS3 (where phi2 = 0)..."', 'text="Fitting εTS₃ (where φ₂ = 0)..."')
content = content.replace('text="Fitting epsTS23 (using all data)..."', 'text="Fitting εTS₂₃ (using all data)..."')

# Success texts
content = content.replace('Successfully fitted epsTS:', 'Successfully fitted εTS:')
content = content.replace('Successfully fitted eps2:', 'Successfully fitted ε₂:')
content = content.replace('Successfully fitted eps3:', 'Successfully fitted ε₃:')
content = content.replace('Successfully fitted eps23:', 'Successfully fitted ε₂₃:')
content = content.replace('Successfully fitted epsTS2:', 'Successfully fitted εTS₂:')
content = content.replace('Successfully fitted epsTS3:', 'Successfully fitted εTS₃:')
content = content.replace('Successfully fitted epsTS23:', 'Successfully fitted εTS₂₃:')

# Select slice paths
content = content.replace('["Constant phi3", "Constant phi2", "Diagonal (phi2 = phi3)"]', '["Constant φ₃", "Constant φ₂", "Diagonal (φ₂ = φ₃)"]')
content = content.replace('== "Constant phi3"', '== "Constant φ₃"')
content = content.replace('== "Constant phi2"', '== "Constant φ₂"')

# Slide selections
content = content.replace('"Select constant phi3 value"', '"Select constant φ₃ value"')
content = content.replace('"Enter constant phi3 value"', '"Enter constant φ₃ value"')
content = content.replace('"Select phi3 from experimental data"', '"Select φ₃ from experimental data"')
content = content.replace('"Select constant phi2 value"', '"Select constant φ₂ value"')
content = content.replace('"Enter constant phi2 value"', '"Enter constant φ₂ value"')
content = content.replace('"Select phi2 from experimental data"', '"Select φ₂ from experimental data"')

# Ternary presets
content = content.replace('"phiS (Contours of subdomain concentrations)"', '"φS (Contours of subdomain concentrations)"')
content = content.replace('"mus2 (Contours of subdomain 2 chemical potentials)"', '"μS2 (Contours of subdomain 2 chemical potentials)"')
content = content.replace('"mus3 (Contours of subdomain 3 chemical potentials)"', '"μS3 (Contours of subdomain 3 chemical potentials)"')
content = content.replace('"ΔΔG_mu (Contours of ΔΔG chemical potentials)"', '"ΔΔG_μ (Contours of ΔΔG chemical potentials)"')
content = content.replace('"Gamma (Contours of preferential interaction coefficients)"', '"Γ (Contours of preferential interaction coefficients)"')
content = content.replace('"Gamma_mu (Contours of preferential interaction mu)"', '"Γ_μ (Contours of preferential interaction μ)"')

# Preset if branches
content = content.replace('elif "phiS" in preset_plot:', 'elif "φS" in preset_plot:')
content = content.replace('elif "mus2" in preset_plot:', 'elif "μS2" in preset_plot:')
content = content.replace('elif "mus3" in preset_plot:', 'elif "μS3" in preset_plot:')
content = content.replace('elif "ΔΔG_mu" in preset_plot:', 'elif "ΔΔG_μ" in preset_plot:')
content = content.replace('elif "Gamma_mu" in preset_plot:', 'elif "Γ_μ" in preset_plot:')
content = content.replace('elif "Gamma" in preset_plot:', 'elif "Γ" in preset_plot:')

# Properties dictionary Gamma
content = content.replace('"Preferential Interaction 2 (Gamma_2)"', '"Preferential Interaction 2 (Γ₂)"')
content = content.replace('"Preferential Interaction 3 (Gamma_3)"', '"Preferential Interaction 3 (Γ₃)"')
content = content.replace('"Preferential Interaction 1,3 (Gamma_1_3)"', '"Preferential Interaction 1,3 (Γ₁,₃)"')

# Concentration axis type mapping
content = re.sub(
    r'(conc_type_plot = st\.selectbox\("Concentration axis type", \["phi", "molar", "molal"\], key="bin_plot_conc"\))',
    r'conc_type_display = st.selectbox("Concentration axis type", ["φ", "molar", "molal"], key="bin_plot_conc")\n                conc_type_plot = "phi" if conc_type_display == "φ" else conc_type_display',
    content
)

with open("app/app.py", "w", encoding="utf-8") as f:
    f.write(content)
