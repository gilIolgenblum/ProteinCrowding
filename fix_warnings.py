import re

with open("app/app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix defaults for ternary to match the explicit value args
content = content.replace('"chi12":     0.1,  "chiTS12":    -0.05', '"chi12":     0.4,  "chiTS12":    4.0')
content = content.replace('"chi13":     0.1,  "chiTS13":    -0.05', '"chi13":     0.4,  "chiTS13":    4.0')

# Remove value=... from ternary number_inputs
content = content.replace('step=0.01, value=1.0, min_value=1.0', 'step=0.01, min_value=1.0')
content = content.replace('step=0.01, value=0.4, format="%.4f"', 'step=0.01, format="%.4f"')
content = content.replace('step=0.01, value=4.0, format="%.4f"', 'step=0.01, format="%.4f"')
content = content.replace('step=0.01, value=0.0, format="%.4f"', 'step=0.01, format="%.4f"')

# Fix deprecated st.components.v1.html -> st.html
# Note: warning said replace with st.iframe but actually st.html is the standard replacement for html blobs in newer streamlit if it's raw HTML, or st.components.v1.html -> st.components.v1.iframe? 
# Wait, "Please replace `st.components.v1.html` with `st.iframe`." -> Wait, st.components.v1.html is deprecated.
# We can just change `st.components.v1.html` to `st.components.v1.html` -> actually the warning says `st.components.v1.html` with `st.iframe`. But wait, st.html was introduced. Let's just use st.html if it exists, or st.components.v1.html.
# I will check app.py for `st.components.v1.html`.
