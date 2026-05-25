import re

with open("app/app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'st.' in line and ('dG' in line or 'dH' in line or 'TdS' in line or 'ddG' in line or 'ddH' in line or 'TddS' in line):
        if 'st.session_state' in line: continue
        if "'dG'" in line or '"dG"' in line: continue
        if "'dH'" in line or '"dH"' in line: continue
        if "'TdS'" in line or '"TdS"' in line: continue
        if "err_dG" in line or "err_dH" in line or "err_TdS" in line: continue
        if "exp_ddG" in line or "exp_ddH" in line or "exp_TddS" in line: continue
        if "err_ddG" in line or "err_ddH" in line or "err_TddS" in line: continue
        print(f"{i+1}: {line.strip()}")

