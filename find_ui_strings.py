import re
with open("app/app.py") as f:
    lines = f.readlines()

patterns = ["dG", "dH", "TdS", "ddG", "ddH", "TddS"]
ignore_words = [".csv", "session_state", "df[", "err_", "raw_", "exp_", "fitted_", "format1", "aq16", "met16", "conc2", "conc3"]

for i, line in enumerate(lines):
    if any(p in line for p in patterns):
        if not any(w in line for w in ignore_words):
            # Print if it has quotes
            if '"' in line or "'" in line:
                print(f"{i+1}: {line.strip()}")
