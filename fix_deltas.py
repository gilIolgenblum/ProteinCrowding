import re

with open("app/app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    # Only modify UI string lines
    if 'st.' in line and ('dG' in line or 'dH' in line or 'TdS' in line):
        # We don't want to replace 'dG' (in quotes) because it refers to dataframe columns
        # So we only replace occurrences that are NOT enclosed in single quotes like 'dG'
        # A simple hack: replace them temporarily, then fix the column references
        tmp = line
        tmp = tmp.replace("ternary dG", "ternary ΔG")
        tmp = tmp.replace("ternary dH", "ternary ΔH")
        tmp = tmp.replace("ternary TdS", "ternary TΔS")
        
        # Are there any other occurrences?
        # Let's check for "dG", "dH", "TdS" that are not inside ' '
        # To be safe, just print them so we can manually review
        if tmp != line:
            print(f"{i+1}: {line.strip()}")
            print(f" -> {tmp.strip()}")
        new_lines.append(tmp)
    else:
        new_lines.append(line)

with open("app/app.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

