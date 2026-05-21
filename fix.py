import os

filepath = 'app/app.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = {
    # File upload strings
    '"Single CSV File (concentration, dG, dH, TdS)"': '"Single CSV File (concentration, ΔG, ΔH, TΔS)"',
    '"Separate CSV Files for dG, dH, TdS"': '"Separate CSV Files for ΔG, ΔH, TΔS"',
    'upload_mode == "Single CSV File (concentration, dG, dH, TdS)"': 'upload_mode == "Single CSV File (concentration, ΔG, ΔH, TΔS)"',
    '"Upload dG CSV (concentration, dG)"': '"Upload ΔG CSV (concentration, ΔG)"',
    '"Upload dH CSV (concentration, dH)"': '"Upload ΔH CSV (concentration, ΔH)"',
    '"Upload TdS CSV (concentration, TdS)"': '"Upload TΔS CSV (concentration, TΔS)"',
    
    # Success / Error messages
    '"dG experimental data loaded!"': '"ΔG experimental data loaded!"',
    '"Error reading dG CSV file: {ex}"': '"Error reading ΔG CSV file: {ex}"',
    '"dH experimental data loaded!"': '"ΔH experimental data loaded!"',
    '"Error reading dH CSV file: {ex}"': '"Error reading ΔH CSV file: {ex}"',
    '"TdS experimental data loaded!"': '"TΔS experimental data loaded!"',
    '"Error reading TdS CSV file: {ex}"': '"Error reading TΔS CSV file: {ex}"',
    '"Ternary dG data loaded!"': '"Ternary ΔG data loaded!"',
    '"Ternary dH data loaded!"': '"Ternary ΔH data loaded!"',
    '"Ternary TdS data loaded!"': '"Ternary TΔS data loaded!"',
    
    # Exceptions
    '"No valid non-NaN experimental dG data points to fit."': '"No valid non-NaN experimental ΔG data points to fit."',
    '"Please upload experimental dG data first!"': '"Please upload experimental ΔG data first!"',
    '"No valid non-NaN experimental dH/TdS data points to fit."': '"No valid non-NaN experimental ΔH/TΔS data points to fit."',
    '"Please upload experimental dH and TdS data first!"': '"Please upload experimental ΔH and TΔS data first!"',
    '"No valid non-NaN experimental dG data points with phi3 <= 0.0011 to fit eps2."': '"No valid non-NaN experimental ΔG data points with phi3 <= 0.0011 to fit eps2."',
    '"Please upload experimental Ternary dG data first!"': '"Please upload experimental Ternary ΔG data first!"',
    '"No valid non-NaN experimental dG data points with phi2 <= 0.0011 to fit eps3."': '"No valid non-NaN experimental ΔG data points with phi2 <= 0.0011 to fit eps3."',
    '"No valid non-NaN experimental dG data points to fit eps23."': '"No valid non-NaN experimental ΔG data points to fit eps23."',
    '"No valid non-NaN experimental dH/TdS data points with phi3 <= 0.0011 to fit epsTS2."': '"No valid non-NaN experimental ΔH/TΔS data points with phi3 <= 0.0011 to fit epsTS2."',
    '"Please upload experimental Ternary dH and TdS data first!"': '"Please upload experimental Ternary ΔH and TΔS data first!"',
    '"No valid non-NaN experimental dH/TdS data points with phi2 <= 0.0011 to fit epsTS3."': '"No valid non-NaN experimental ΔH/TΔS data points with phi2 <= 0.0011 to fit epsTS3."',
    '"No valid non-NaN experimental dH/TdS data points to fit epsTS23."': '"No valid non-NaN experimental ΔH/TΔS data points to fit epsTS23."',
    
    # Plot preset names and checks
    '"ddG (3x3 contour)"': '"ΔΔG (3x3 contour)"',
    '"TdS_mix (Contours of mixing entropy)"': '"TΔS_mix (Contours of mixing entropy)"',
    '"dG_mix (Contours of mixing free energy)"': '"ΔG_mix (Contours of mixing free energy)"',
    '"ddG_mu (Contours of ddG chemical potentials)"': '"ΔΔG_mu (Contours of ΔΔG chemical potentials)"',
    '"TddS (Contours of TddS entropy)"': '"TΔΔS (Contours of TΔΔS entropy)"',
    '"ddH (Contours of ddH enthalpy)"': '"ΔΔH (Contours of ΔΔH enthalpy)"',
    '["ddG (3x3 contour)", "TddS", "ddH"]': '["ΔΔG (3x3 contour)", "TΔΔS", "ΔΔH"]',
    'is_G = "ddG" in preset_plot': 'is_G = "ΔΔG" in preset_plot',
    
    # 2D contour options
    '"Free Energy (ddG) [kT]"': '"Free Energy (ΔΔG) [kT]"',
    '"Free Energy (ddG) [kJ]"': '"Free Energy (ΔΔG) [kJ]"',
    '"Enthalpy (ddH) [kT]"': '"Enthalpy (ΔΔH) [kT]"',
    '"Enthalpy (ddH) [kJ]"': '"Enthalpy (ΔΔH) [kJ]"',
    '"Entropy (TddS) [kT]"': '"Entropy (TΔΔS) [kT]"',
    '"Entropy (TddS) [kJ]"': '"Entropy (TΔΔS) [kJ]"',

    # Also fix some UI column texts:
    "Missing required columns for dG. Expected: 'concentration', 'dG'.": "Missing required columns for ΔG. Expected: 'concentration', 'dG'.",
    "Missing required columns for dH. Expected: 'concentration', 'dH'.": "Missing required columns for ΔH. Expected: 'concentration', 'dH'.",
    "Missing required columns for TdS. Expected: 'concentration', 'TdS'.": "Missing required columns for TΔS. Expected: 'concentration', 'TdS'."
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Replaced all strings.")
