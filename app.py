import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from model import (
    simulate_model1_trajectory,
    simulate_model2_trajectory,
    F1_analytical_model1,
    F1_analytical_model2
)

# ----------------------------
# Streamlit configuration
# ----------------------------
st.set_page_config(
    page_title="Quantum Channel and Transduction Fidelity Simulator",
    layout="wide"
)

st.title("Quantum Channel and Transduction Fidelity Simulator")
st.markdown(
    """
    Monte Carlo simulation and analytical comparison with two photonic encoding models 
    
    **Model 1 (Time Bins)** vs **Model 2 (Single Rail Fock States)**
    """
)
# Author info
st.markdown(
    "<sub>Author: Francesco Fiorini — francesco.fiorini@phd.unipi.it</sub>",
    unsafe_allow_html=True
)
# ============================
# Sidebar — Parameters
# ============================
st.sidebar.header("Simulation parameters")

run_mode = st.sidebar.selectbox(
    "Run mode",
    [
        "Model 1 only (Time Bins)",
        "Model 2 only (Fock States)",
        "Both models (comparison)"
    ]
)

F_or_P = st.sidebar.slider(
    "Source success probability",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.05
)

eta_t = st.sidebar.slider(
    "Transduction efficiency ηₜ",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

eta_c_fixed = st.sidebar.slider(
    "Fiber transmissivity η𝚌",
    min_value=0.0,
    max_value=1.0,
    value=0.63,
    step=0.01
)

n_bar = st.sidebar.slider(
    "Thermal photon number n̄",
    min_value=0.0,
    max_value=0.1,
    value=0.01,
    step=0.01
)

dim_photonic = st.sidebar.selectbox(
    "Photonic Hilbert dimension truncation",
    [2, 3, 4, 5, 6],
    index=2
)

num_trials = st.sidebar.selectbox(
    "Monte Carlo trials per point",
    [100, 500, 1000, 5000],
    index=2
)

scan_type = st.sidebar.radio(
    "Scan variable",
    ["ηₜ", "Fiber length", "Thermal noise"]
)

run_button = st.sidebar.button("Run simulation")

# ============================
# Main simulation
# ============================
if run_button:
    st.session_state["ran"] = True

    progress = st.progress(0.0)
    status = st.empty()

    # Define scan axis
    if scan_type == "ηₜ":
        x_vals = np.linspace(0, 1, 20)
        xlabel = r"Transduction efficiency $\eta_t$"

    elif scan_type == "Fiber length":
        x_vals = np.linspace(0, 50, 20)
        xlabel = r"Fiber length $l$ (km)"

    else:
        x_vals = np.linspace(0, 0.5, 20)
        xlabel = r"Mean thermal photon number $\bar{n}$"

    fidel_sim_m1, fidel_anal_m1 = [], []
    fidel_sim_m2, fidel_anal_m2 = [], []

    # ------------------------
    # Main loop
    # ------------------------
    for i, x in enumerate(x_vals):

        if scan_type == "ηₜ":
            eta_t_i = x
            eta_c_i = eta_c_fixed
            n_bar_i = n_bar

        elif scan_type == "Fiber length":
            eta_t_i = eta_t
            eta_c_i = np.exp(-x / 22)
            n_bar_i = n_bar

        else:
            eta_t_i = eta_t
            eta_c_i = eta_c_fixed
            n_bar_i = x

        # ===== Model 1 =====
        if run_mode in ["Model 1 only (Time Bins)", "Both models (comparison)"]:
            sims_m1 = [
                simulate_model1_trajectory(
                    F_or_P, eta_c_i, eta_t_i, n_bar_i, dim_photonic
                )[0]
                for _ in range(num_trials)
            ]
            fidel_sim_m1.append(np.mean(sims_m1))
            fidel_anal_m1.append(
                F1_analytical_model1(
                    eta_t_i, F_or_P, eta_c_i, n_bar_i
                )
            )

        # ===== Model 2 =====
        if run_mode in ["Model 2 only (Single Rail)", "Both models (comparison)"]:
            sims_m2 = [
                simulate_model2_trajectory(
                    F_or_P, eta_c_i, eta_t_i, n_bar_i, dim_photonic
                )[0]
                for _ in range(num_trials)
            ]
            fidel_sim_m2.append(np.mean(sims_m2))
            fidel_anal_m2.append(
                F1_analytical_model2(
                    eta_t_i, F_or_P, eta_c_i, n_bar_i
                )
            )

        progress.progress((i + 1) / len(x_vals))
        status.text(f"Running point {i + 1} / {len(x_vals)}")

    # ============================
    # Prepare results for CSV export
    # ============================
    scan_type_csv_map = {
        "ηₜ": "Transduction efficiency",
        "Fiber length": "Fiber length",
        "Thermal noise": "Thermal noise"
    }

    scan_type_csv = scan_type_csv_map[scan_type]

    data = {
        "scan_variable": [f"{x:.4f}".replace(".", ",") for x in x_vals],
        "scan_type": [scan_type_csv] * len(x_vals),
        "run_mode": [run_mode] * len(x_vals),
        "F_or_P": [F_or_P] * len(x_vals),
        "eta_t_fixed": [eta_t] * len(x_vals),
        "eta_c_fixed": [eta_c_fixed] * len(x_vals),
        "n_bar_fixed": [n_bar] * len(x_vals),
        "dim_photonic": [dim_photonic] * len(x_vals),
        "num_trials": [num_trials] * len(x_vals),
    }

    if fidel_sim_m1:
        data["model1_simulation"] = fidel_sim_m1
        data["model1_analytical"] = fidel_anal_m1

    if fidel_sim_m2:
        data["model2_simulation"] = fidel_sim_m2
        data["model2_analytical"] = fidel_anal_m2

    results_df = pd.DataFrame(data)

    st.session_state["results_df"] = results_df
    st.session_state["plot_data"] = (x_vals, fidel_sim_m1, fidel_anal_m1,
                                     fidel_sim_m2, fidel_anal_m2, xlabel)

# ============================
# Display results (persistent)
# ============================
if st.session_state.get("ran", False):

    x_vals, fidel_sim_m1, fidel_anal_m1, fidel_sim_m2, fidel_anal_m2, xlabel = (
        st.session_state["plot_data"]
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    if fidel_sim_m1:
        ax.plot(x_vals, fidel_sim_m1, "o-", linewidth=3, label="Model 1 – Simulation")
        ax.plot(x_vals, fidel_anal_m1, "--", linewidth=3, label="Model 1 – Analytical")

    if fidel_sim_m2:
        ax.plot(x_vals, fidel_sim_m2, "s-", linewidth=3, label="Model 2 – Simulation")
        ax.plot(x_vals, fidel_anal_m2, "--", linewidth=3, label="Model 2 – Analytical")

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(r"Fidelity $F$", fontsize=18)
    ax.grid(True)
    ax.legend(fontsize=13)

    st.pyplot(fig)

    st.subheader("Average fidelities")

    if fidel_sim_m1:
        st.write(f"**Model 1 – Simulation:** {np.mean(fidel_sim_m1):.4f}")
        st.write(f"**Model 1 – Analytical:** {np.mean(fidel_anal_m1):.4f}")

    if fidel_sim_m2:
        st.write(f"**Model 2 – Simulation:** {np.mean(fidel_sim_m2):.4f}")
        st.write(f"**Model 2 – Analytical:** {np.mean(fidel_anal_m2):.4f}")

    # ============================
    # CSV Export
    # ============================
    st.markdown("### 📥 Export simulation results")

    csv = st.session_state["results_df"].to_csv(index=False, sep=";").encode("utf-8")

    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name="quantum_transduction_results.csv",
        mime="text/csv",
    )
