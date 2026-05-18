# Antigravity Project Plan: FH Crowding Research Tool

## Project Goal

Convert the current FH crowding/synergy analysis code from a single Python file plus demonstration notebooks into a reusable, reproducible, peer-friendly scientific software project.

The intended users are researchers who may want to:

- Run the model from Python scripts.
- Reproduce published or shared notebook analyses.
- Upload their own data through a simple local web app.
- Fit model parameters.
- Export plots and fitted results.

The preferred development path is:

```text
Python package first
↓
Clean example notebooks
↓
Local Streamlit app
↓
Optional hosted web app later
```

Do not start with a full hosted website or desktop GUI until the core package API is stable.

---

## Current Project State

The project currently contains:

- One main Python source file containing the core scientific model.
- One or more Jupyter notebooks demonstrating usage.
- Model classes for cosolutes, proteins, binary crowding, and ternary cosolute mixtures.
- Fitting routines for thermodynamic parameters.
- Plotting functions.
- Pandas export routines.

Treat the existing file as the scientific reference implementation unless tests or user instructions indicate otherwise.

---

## Highest-Level Rule

Preserve scientific meaning.

When refactoring or extending the code, do not change equations, units, signs, parameter definitions, fitting logic, or thermodynamic interpretations unless explicitly requested.

If a change could alter numerical results, flag it clearly before applying it.

---

## Recommended Repository Structure

Refactor toward this structure:

```text
fh-crowding/
├── pyproject.toml
├── README.md
├── LICENSE
├── CITATION.cff
├── src/
│   └── fh_crowding/
│       ├── __init__.py
│       ├── constants.py
│       ├── protein.py
│       ├── cosolute.py
│       ├── binary.py
│       ├── ternary.py
│       ├── fitting.py
│       ├── plotting.py
│       ├── io.py
│       └── validation.py
├── examples/
│   ├── 01_binary_fit.ipynb
│   ├── 02_ternary_fit.ipynb
│   └── 03_export_results.ipynb
├── app/
│   ├── app.py
│   └── sample_data/
├── tests/
│   ├── test_binary_model.py
│   ├── test_ternary_model.py
│   ├── test_fitting.py
│   ├── test_units.py
│   └── test_io.py
└── docs/
    └── model_notes.md
```

Do not move everything at once unless explicitly requested.

Prefer incremental, reviewable changes.

---

## Development Priorities

### 1. Stabilize the Core Package

First, make the model importable as a package:

```python
from fh_crowding import Protein, Cosolute, BinaryCrowdingModel, TernaryCrowdingModel
```

The package should support local editable installation:

```bash
pip install -e .
```

The package should not require notebooks to run.

### 2. Define a Clean Public API

Expose simple user-facing classes and functions.

The target API should look approximately like:

```python
from fh_crowding import Protein, Cosolute, BinaryCrowdingModel

protein = Protein(sasa=419.0)

urea = Cosolute(
    nu=...,
    chi=...,
    chi_ts=...
)

model = BinaryCrowdingModel(
    protein=protein,
    cosolute=urea,
    eps=0.0,
    eps_ts=0.0,
    temperature=298.15
)

model.solve()
results = model.to_dataframe()
fit = model.fit_eps(exp_conc, exp_ddg, concentration_type="molal")
```

Keep internal helper functions private when appropriate.

Avoid exposing implementation details unless they are scientifically meaningful outputs.

### 3. Preserve Reproducible Notebooks

The notebooks should become clean examples, not the core implementation.

Each example notebook should include:

1. Scientific question.
2. Input data.
3. Parameter definitions.
4. Model construction.
5. Fit or simulation.
6. Diagnostic plots.
7. Exported results.
8. Short interpretation.

Do not duplicate model code inside notebooks.

Notebooks should import from the package.

### 4. Add a Local Streamlit App

After the package API is stable, create a local Streamlit app.

The app should allow users to:

- Select binary or ternary model.
- Input protein SASA.
- Input FH parameters.
- Input soft interaction parameters.
- Upload CSV data.
- Select concentration type: `phi`, `molar`, or `molal`.
- Run fits.
- View fitted parameters.
- View model plots.
- Download model results as CSV.
- Download figures.

The Streamlit app should import the package. It should not duplicate scientific logic.

### 5. Consider Hosting Only Later

Do not build a public hosted website until the local package and Streamlit app are stable.

Before hosting, consider:

- Uploaded data privacy.
- File validation.
- Runtime limits.
- Server cost.
- Authentication.
- Version tracking.
- Citation and licensing.
- Reproducibility.

---

## Scientific Rules

### Units

Always track units explicitly.

Common quantities include:

- Temperature in K.
- Energies in kJ/mol or kcal/mol.
- Volume fractions as dimensionless `phi`.
- Molar concentration in mol/L.
- Molality in mol/kg.
- Osmotic pressure or osmotic concentration according to the model definition.

Never silently convert units.

If code converts units, make the conversion obvious in the function name, docstring, or variable name.

### Thermodynamic Meaning

Distinguish clearly between:

- Free energy.
- Enthalpy.
- Entropy contribution.
- `TΔS`.
- Folding and unfolding conventions.
- Dimensionless model quantities.
- kJ/mol values.
- kcal/mol values.

Be especially careful with sign conventions.

Do not rename thermodynamic variables unless the mapping is clear.

### Model Parameters

Preserve the definitions of:

- `nu`
- `chi`
- `chiTS`
- `chiH`
- `eps`
- `epsTS`
- `epsH`
- `eps23`
- `epsTS23`
- `SASA`
- `phiC`
- `phiS`
- `phi1`, `phi2`, `phi3`

If renaming variables for readability, maintain backwards compatibility or provide a clear migration path.

### Numerical Solvers

When editing solver code:

- Preserve initial guesses unless improving them intentionally.
- Check convergence.
- Store solver diagnostics.
- Avoid hiding failed solves.
- Avoid silently replacing invalid values with plausible-looking values.
- Avoid broad exception handling that hides numerical problems.

If a solve fails, the code should report the concentration point and relevant parameters.

---

## Refactoring Rules

### General

Refactor incrementally.

Prefer small, safe patches over large rewrites.

Do not change numerical behavior unless requested.

When moving code into modules, preserve function and class behavior.

After each meaningful refactor, run a minimal numerical comparison against the original code.

### Class Naming

Current names may be scientifically meaningful but should be made clearer for users.

Potential public-facing names:

```text
protein          -> Protein
cosolute         -> Cosolute
cosolutes        -> CosoluteMixture
crowding         -> BinaryCrowdingModel
crowding_ter     -> TernaryCrowdingModel
```

If renaming, consider maintaining aliases for backwards compatibility.

### Function Naming

Prefer clear names for public methods:

```text
solve_equil()          -> solve_equilibrium()
to_pandas()            -> to_dataframe()
plot_results()         -> plot_results()
fit_eps()              -> fit_eps()
fit_epsTS()            -> fit_eps_ts()
```

Do not rename methods in a way that breaks existing notebooks unless the notebooks are updated in the same change.

### Data Mutability

Avoid surprising mutation of model state when possible.

Fitting methods should make it clear when they update model parameters.

Prefer returning fit result objects or dictionaries containing:

- fitted parameters
- objective value
- convergence status
- optimizer message
- number of function evaluations
- predicted model values

---

## Testing Rules

Add tests before or alongside major refactors.

Minimum tests:

### Import Test

Verify the package imports cleanly:

```python
import fh_crowding
```

### Smoke Tests

Create simple binary and ternary models and verify that:

- Solving completes.
- Result arrays have expected shapes.
- Output dataframes contain expected columns.
- Values are finite over valid concentration ranges.

### Unit Conversion Tests

Check conversions among:

- dimensionless model units
- kJ/mol
- kcal/mol
- molar
- molal
- volume fraction

### Synthetic Fit Tests

Generate synthetic data from known parameters, add no noise or small noise, and test whether the fitting recovers parameters approximately.

### Regression Tests

For known example inputs, compare key outputs against reference values.

Do not update reference values casually. If reference values change, explain why.

---

## Plotting Rules

All plots should:

- Label axes.
- Include units where applicable.
- Distinguish model curves from experimental points.
- Indicate whether values are folding or unfolding quantities.
- Avoid hidden normalization or smoothing.
- Preserve raw data values unless transformation is explicitly requested.

For publication-quality figures, expose figure and axes objects when possible.

Do not call `plt.show()` inside low-level plotting functions unless requested. Prefer returning figure objects.

Recommended pattern:

```python
fig, ax = plot_binary_fit(...)
return fig, ax
```

---

## I/O Rules

Input data should preferably be CSV.

Validate uploaded or loaded data before running fits.

Required columns should be explicit.

For example:

```text
concentration
ddG
err_ddG
ddH
err_ddH
TddS
err_TddS
```

Use tolerant but explicit column mapping in user-facing tools.

Do not assume column names silently.

For Streamlit, show users what columns were detected.

---

## Streamlit App Rules

The app should be simple and scientific, not flashy.

Recommended pages:

```text
1. Binary Model
2. Ternary Model
3. Fit Experimental Data
4. Download Results
5. Documentation
```

The app should always display:

- Input parameters.
- Fitted parameters.
- Solver or optimizer status.
- Plots.
- Download buttons for CSV results.
- Download buttons for figures.

The app should never overwrite files automatically.

The app should not store uploaded user data unless explicitly designed to do so.

---

## Documentation Rules

The README should include:

1. What the package does.
2. Installation instructions.
3. Minimal working example.
4. Example notebook links.
5. Input data format.
6. Main model assumptions.
7. Citation information.
8. License.
9. Contact or issue-reporting instructions.

The documentation should be honest about model assumptions and limitations.

Do not overstate generality.

---

## Packaging Rules

Use `pyproject.toml`.

Prefer a `src/` layout.

Suggested dependencies:

```text
numpy
pandas
matplotlib
scipy
```

Optional dependencies:

```text
streamlit
jupyter
pytest
```

Separate core dependencies from app/dev dependencies.

Example extras:

```toml
[project.optional-dependencies]
app = ["streamlit"]
dev = ["pytest", "jupyter"]
```

Do not add unnecessary dependencies.

---

## Git Rules

Before editing, inspect repository state.

Do not commit, push, reset, clean, rebase, or delete files unless explicitly requested.

Prefer small commits or patches grouped by purpose:

```text
1. Package skeleton
2. Move constants and base classes
3. Move binary model
4. Move ternary model
5. Add tests
6. Clean notebooks
7. Add Streamlit app
```

---

## Workflow: Package the Project

Use this workflow when asked to convert the current code into a package.

### Steps

1. Inspect current files.
2. Identify public classes and methods.
3. Create package skeleton.
4. Move code incrementally.
5. Preserve backward compatibility where practical.
6. Add imports in `__init__.py`.
7. Add `pyproject.toml`.
8. Add a minimal README.
9. Add smoke tests.
10. Verify that notebooks can import the package.

### Do Not

- Rewrite equations.
- Change units.
- Change signs.
- Remove existing functionality.
- Break notebooks without updating them.

### Final Report

Include:

- Files changed.
- Public API added.
- Compatibility notes.
- Tests run.
- Remaining work.

---

## Workflow: Build the Streamlit App

Use this workflow when asked to build the local app.

### Steps

1. Confirm the package imports correctly.
2. Create `app/app.py`.
3. Add sidebar inputs for model parameters.
4. Add CSV upload.
5. Validate data columns.
6. Run model or fit.
7. Show plots.
8. Show fitted parameters.
9. Provide CSV download.
10. Provide figure download if practical.

### Do Not

- Duplicate model logic.
- Store uploaded files permanently.
- Hide failed fits.
- Silently transform data.
- Make the app depend on notebooks.

### Final Report

Include:

- How to run the app.
- Required input format.
- Files added.
- Known limitations.

---

## Workflow: Clean Example Notebooks

Use this workflow when asked to clean notebooks.

### Steps

1. Preserve the scientific analysis.
2. Remove exploratory dead code.
3. Replace local imports with package imports.
4. Add explanatory markdown.
5. Make file paths configurable.
6. Ensure the notebook runs from top to bottom.
7. Export important results to CSV.
8. Keep figures reproducible.

### Do Not

- Change scientific conclusions.
- Hide preprocessing.
- Hardcode private absolute paths.
- Duplicate package code.

---

## Workflow: Scientific Code Review

Use this workflow when asked to review the code.

### Review Categories

1. Numerical correctness.
2. Thermodynamic consistency.
3. Unit handling.
4. Solver robustness.
5. Fitting robustness.
6. API clarity.
7. Reproducibility.
8. Plotting and export behavior.
9. Documentation gaps.
10. Test coverage.

### Output Format

Use this structure:

```text
Critical issues
Important improvements
Minor improvements
Recommended next steps
```

---

## Near-Term Roadmap

### Phase 1: Make It Installable

- Add package skeleton.
- Add `pyproject.toml`.
- Add `README.md`.
- Expose a minimal public API.
- Confirm examples still run.

### Phase 2: Make It Trustworthy

- Add tests.
- Add solver diagnostics.
- Add input validation.
- Add regression examples.

### Phase 3: Make It Usable

- Clean notebooks.
- Add sample datasets.
- Add better plotting functions.
- Add simple documentation.

### Phase 4: Make It Accessible

- Add local Streamlit app.
- Allow CSV upload.
- Add downloads.
- Add a short user guide.

### Phase 5: Decide on Hosting

Only after collaborators test the local version, decide whether to host the app publicly.

---

## Default Decision

When uncertain, choose the option that improves:

1. Scientific reproducibility.
2. API clarity.
3. Testability.
4. Maintainability.
5. Ease of use for researchers.

Avoid premature GUI complexity.
