import re

def main():
    with open("app/app.py", "r") as f:
        content = f.read()

    # We will find the assignments:
    # log_runtime_state("after fit optimization")
    # fit_progress.progress(1.0, text="Fit completed!")
    # [optional space/comments]
    # st.session_state["fitted_XYZ"] = model.XYZ
    # st.session_state["fit_updated"] = True
    # st.session_state["is_fitting_mode"] = True

    pattern = re.compile(
        r'log_runtime_state\("after fit optimization"\)\s*'
        r'fit_progress\.progress\(1\.0,\s*text="Fit completed!"\)\s*'
        r'(?:# Save state \(lightweight only\)\s*)?'
        r'(st\.session_state\["fitted_[a-zA-Z0-9]+"\] = model\.[a-zA-Z0-9]+)\s*'
        r'st\.session_state\["fit_updated"\] = True\s*'
        r'st\.session_state\["is_fitting_mode"\] = True'
    )

    def replacer(m):
        fitted_assign = m.group(1)
        
        # Determine if binary or ternary based on the parameter fitted (eps/epsTS vs eps2/eps3/eps23)
        if "fitted_eps\"" in fitted_assign or "fitted_epsTS\"" in fitted_assign:
            args = "model_type, model, protein, cosolute, T, phiC_max"
        else:
            args = "model_type, model, protein, cosolutes, T, None, phi2_max, phi3_max"

        return (
            f'log_runtime_state("after fit optimization")\n'
            f'                            {fitted_assign}\n'
            f'                            \n'
            f'                            fit_progress.progress(0.8, text="Generating minimal grid preview...")\n'
            f'                            try:\n'
            f'                                preview_model = run_minimal_simulation_preview({args})\n'
            f'                                st.session_state["solved_model"] = preview_model\n'
            f'                                st.session_state["solved_model_type"] = model_type\n'
            f'                            except Exception as e:\n'
            f'                                logger.warning(f"Failed to generate minimal preview: {{e}}")\n'
            f'                                \n'
            f'                            st.session_state["fit_updated"] = True\n'
            f'                            st.session_state["is_fitting_mode"] = True\n'
            f'                            fit_progress.progress(1.0, text="Fit & Preview updated!")'
        )

    new_content = pattern.sub(replacer, content)

    # Also, update the UI button string:
    new_content = new_content.replace(
        'st.info("The fitted parameters have been updated. Run the full grid simulation to visualize the results.")',
        'st.info("A minimal grid preview is shown below. Run the full simulation to calculate the denser grid.")'
    )
    new_content = new_content.replace(
        'if st.button("🚀 Run simulation with fitted parameters", use_container_width=True):',
        'if st.button("🚀 Run full simulation with fitted parameters", use_container_width=True):'
    )

    with open("app/app.py", "w") as f:
        f.write(new_content)

if __name__ == "__main__":
    main()
