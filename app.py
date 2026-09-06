import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from coupler_engine import run_simulation


st.set_page_config(
    page_title="Directional Coupler Simulator",
    layout="wide",
)


def run_gap_sweep(
    w_single,
    h_core,
    gap_start,
    gap_end,
    n_gap,
    coupler_L,
    ring_R,
    wavelength_ref,
    polarization,
    res_mode,
    top_oxide,
    bottom_oxide,
):
    """Run the simulation for a range of gaps at one reference wavelength."""

    if gap_end <= gap_start:
        raise ValueError("Gap End must be greater than Gap Start.")

    gaps = np.linspace(gap_start, gap_end, n_gap)

    output = {
        "scan_mode": "Gap",
        "gap_vec": [],
        "kappa_vec": [],
        "p_cross_vec": [],
        "p_bar_vec": [],
        "l_residual_vec": [],
        "l_total_vec": [],
        "neff_even": [],
        "neff_odd": [],
        "lambda_ref": wavelength_ref,
    }

    # The engine may not support n_lambda=1.
    # A very narrow 3-point wavelength range is therefore used.
    delta_lambda = 1e-6

    for current_gap in gaps:
        result = run_simulation(
            w_single=w_single,
            h_core=h_core,
            gap=float(current_gap),
            coupler_L=coupler_L,
            ring_R=ring_R,
            lambda_start=wavelength_ref - delta_lambda,
            lambda_end=wavelength_ref + delta_lambda,
            n_lambda=3,
            polarization=polarization,
            res_mode=res_mode,
            top_oxide=top_oxide,
            bottom_oxide=bottom_oxide,
        )

        lambda_vec = np.asarray(result["lambda_vec"])
        index = int(np.argmin(np.abs(lambda_vec - wavelength_ref)))

        output["gap_vec"].append(float(current_gap))
        output["kappa_vec"].append(result["kappa_vec"][index])
        output["p_cross_vec"].append(result["p_cross_vec"][index])
        output["p_bar_vec"].append(result["p_bar_vec"][index])
        output["l_residual_vec"].append(result["l_residual_vec"][index])
        output["l_total_vec"].append(result["l_total_vec"][index])
        output["neff_even"].append(result["neff_even"][index])
        output["neff_odd"].append(result["neff_odd"][index])

    for key in output:
        if key != "scan_mode" and key != "lambda_ref":
            output[key] = np.asarray(output[key])

    return output


def calculate_wavelength_results(results, losses):
    """Calculate loss and Q values for wavelength sweep."""

    alpha_db = np.asarray(losses, dtype=float)
    alpha_cm = alpha_db * (np.log(10.0) / 10.0)

    ring_length_cm = results["L_ring_um"] * 1e-4
    round_trip_loss_pct = (
        1.0 - np.exp(-alpha_cm * ring_length_cm)
    ) * 100.0

    lambda_vec = np.asarray(results["lambda_vec"])
    neff_even = np.asarray(results["neff_even"])
    neff_odd = np.asarray(results["neff_odd"])
    neff_average = (neff_even + neff_odd) / 2.0

    center_index = int(
        results.get("idx_center", len(lambda_vec) // 2)
    )

    lambda_center_um = float(
        results.get("lambda_center_val", lambda_vec[center_index])
    )
    lambda_center_cm = lambda_center_um * 1e-4

    if len(lambda_vec) > 1 and lambda_vec[-1] != lambda_vec[0]:
        dneff_dlambda = (
            neff_average[-1] - neff_average[0]
        ) / ((lambda_vec[-1] - lambda_vec[0]) * 1e-4)

        n_group = (
            neff_average[center_index]
            - lambda_center_cm * dneff_dlambda
        )
    else:
        n_group = neff_average[center_index]

    q0_values = (
        2.0 * np.pi * n_group
    ) / (lambda_center_cm * alpha_cm)

    results["alpha_db_vals"] = alpha_db
    results["round_trip_loss_pct"] = round_trip_loss_pct
    results["QL_vals"] = q0_values / 2.0

    return results


def create_wavelength_dataframe(data):
    return pd.DataFrame(
        {
            "Wavelength_um": data["lambda_vec"],
            "Neff_Even": data["neff_even"],
            "Neff_Odd": data["neff_odd"],
            "Kappa_1_per_um": data["kappa_vec"],
            "L_residual_um": data["l_residual_vec"],
            "L_total_um": data["l_total_vec"],
            "P_cross_percent": data["p_cross_vec"],
            "P_bar_percent": data["p_bar_vec"],
        }
    )


def create_gap_dataframe(data):
    return pd.DataFrame(
        {
            "Gap_um": data["gap_vec"],
            "Neff_Even": data["neff_even"],
            "Neff_Odd": data["neff_odd"],
            "Kappa_1_per_um": data["kappa_vec"],
            "L_residual_um": data["l_residual_vec"],
            "L_total_um": data["l_total_vec"],
            "P_cross_percent": data["p_cross_vec"],
            "P_bar_percent": data["p_bar_vec"],
        }
    )


def display_gap_results(data):
    gaps = data["gap_vec"]
    kappa = data["kappa_vec"]
    p_cross = data["p_cross_vec"]
    p_bar = data["p_bar_vec"]

    st.title("📈 Gap Sweep Analysis")

    st.markdown(
        f"Scanning gap from **{gaps[0]:.3f} μm** to "
        f"**{gaps[-1]:.3f} μm** at "
        f"λ = **{data['lambda_ref']:.3f} μm**"
    )

    best_cross_index = int(np.argmax(p_cross))

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Maximum κ",
        f"{np.max(kappa):.4f} μm⁻¹",
    )

    col2.metric(
        "Maximum Cross Power",
        f"{np.max(p_cross):.2f} %",
    )

    col3.metric(
        "Gap at Maximum Cross Power",
        f"{gaps[best_cross_index]:.3f} μm",
    )

    st.subheader("Coupling Coefficient vs. Gap")

    fig_kappa, ax_kappa = plt.subplots(figsize=(8, 5))
    ax_kappa.plot(
        gaps,
        kappa,
        "o-",
        color="darkorange",
        linewidth=2,
    )
    ax_kappa.set_xlabel("Gap [μm]")
    ax_kappa.set_ylabel("κ [μm⁻¹]")
    ax_kappa.set_title("Coupling Coefficient as a Function of Gap")
    ax_kappa.grid(True)
    st.pyplot(fig_kappa)
    plt.close(fig_kappa)

    st.subheader("Power Transfer vs. Gap")

    fig_power, ax_power = plt.subplots(figsize=(8, 5))
    ax_power.plot(
        gaps,
        p_cross,
        "o-",
        color="royalblue",
        linewidth=2,
        label="Cross Port",
    )
    ax_power.plot(
        gaps,
        p_bar,
        "s-",
        color="forestgreen",
        linewidth=2,
        label="Bar Port",
    )
    ax_power.set_xlabel("Gap [μm]")
    ax_power.set_ylabel("Power [%]")
    ax_power.set_ylim(0, 105)
    ax_power.set_title("Power Transfer as a Function of Gap")
    ax_power.grid(True)
    ax_power.legend()
    st.pyplot(fig_power)
    plt.close(fig_power)

    st.subheader("Effective Index vs. Gap")

    fig_neff, ax_neff = plt.subplots(figsize=(8, 5))
    ax_neff.plot(
        gaps,
        data["neff_even"],
        "o-",
        label="Even Mode",
    )
    ax_neff.plot(
        gaps,
        data["neff_odd"],
        "s-",
        label="Odd Mode",
    )
    ax_neff.set_xlabel("Gap [μm]")
    ax_neff.set_ylabel("Effective Index")
    ax_neff.set_title("Effective Index as a Function of Gap")
    ax_neff.grid(True)
    ax_neff.legend()
    st.pyplot(fig_neff)
    plt.close(fig_neff)

    st.subheader("Gap Sweep Data")

    gap_df = create_gap_dataframe(data)
    st.dataframe(gap_df, use_container_width=True)

    csv_data = gap_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📄 Download Gap Sweep CSV",
        data=csv_data,
        file_name="gap_sweep_results.csv",
        mime="text/csv",
        use_container_width=True,
    )


def display_wavelength_results(data):
    lambda_vec = np.asarray(data["lambda_vec"])
    center_index = int(
        data.get("idx_center", len(lambda_vec) // 2)
    )

    st.title("📈 Wavelength Sweep Analysis")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Central κ",
        f"{data['kappa_vec'][center_index]:.4f} μm⁻¹",
    )

    col2.metric(
        "Residual Length",
        f"{data['l_residual_vec'][center_index]:.2f} μm",
    )

    col3.metric(
        "Cross Power",
        f"{data['p_cross_vec'][center_index]:.2f} %",
    )

    col4.metric(
        "Central Wavelength",
        f"{lambda_vec[center_index]:.3f} μm",
    )

    st.subheader("Effective Index vs. Wavelength")

    fig_neff, ax_neff = plt.subplots(figsize=(8, 5))
    ax_neff.plot(
        lambda_vec,
        data["neff_even"],
        "o-",
        label="Even Mode",
    )
    ax_neff.plot(
        lambda_vec,
        data["neff_odd"],
        "s-",
        label="Odd Mode",
    )
    ax_neff.set_xlabel("Wavelength [μm]")
    ax_neff.set_ylabel("Effective Index")
    ax_neff.set_title("Supermode Dispersion")
    ax_neff.grid(True)
    ax_neff.legend()
    st.pyplot(fig_neff)
    plt.close(fig_neff)

    st.subheader("Coupling Coefficient and Residual Length")

    fig_kappa, ax_kappa = plt.subplots(figsize=(8, 5))
    ax_kappa.plot(
        lambda_vec,
        data["kappa_vec"],
        "o-",
        color="darkorange",
        label="κ",
    )

    ax_kappa.set_xlabel("Wavelength [μm]")
    ax_kappa.set_ylabel("κ [μm⁻¹]", color="darkorange")
    ax_kappa.grid(True)

    ax_length = ax_kappa.twinx()
    ax_length.plot(
        lambda_vec,
        data["l_residual_vec"],
        "s--",
        color="purple",
        label="Residual Length",
    )
    ax_length.set_ylabel(
        "Residual Length [μm]",
        color="purple",
    )

    ax_kappa.set_title("Coupling Parameters vs. Wavelength")
    st.pyplot(fig_kappa)
    plt.close(fig_kappa)

    st.subheader("Power Transfer vs. Wavelength")

    fig_power, ax_power = plt.subplots(figsize=(8, 5))
    ax_power.plot(
        lambda_vec,
        data["p_cross_vec"],
        "o-",
        label="Cross Port",
    )
    ax_power.plot(
        lambda_vec,
        data["p_bar_vec"],
        "s-",
        label="Bar Port",
    )
    ax_power.set_xlabel("Wavelength [μm]")
    ax_power.set_ylabel("Power [%]")
    ax_power.set_ylim(0, 105)
    ax_power.set_title("Power Transfer")
    ax_power.grid(True)
    ax_power.legend()
    st.pyplot(fig_power)
    plt.close(fig_power)

    st.subheader("Wavelength Sweep Data")

    wavelength_df = create_wavelength_dataframe(data)
    st.dataframe(wavelength_df, use_container_width=True)

    csv_data = wavelength_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📄 Download Wavelength Sweep CSV",
        data=csv_data,
        file_name="wavelength_sweep_results.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.header("🛠️ Coupler Parameters")

w_single = st.sidebar.number_input(
    "Waveguide Width w [μm]",
    min_value=0.01,
    value=1.0,
    step=0.1,
)

h_core = st.sidebar.number_input(
    "Waveguide Height h [μm]",
    min_value=0.01,
    value=0.3,
    step=0.05,
)

gap = st.sidebar.number_input(
    "Coupler Gap [μm]",
    min_value=0.001,
    value=0.3,
    step=0.05,
)

coupler_L = st.sidebar.number_input(
    "Straight Length L [μm]",
    min_value=0.01,
    value=35.0,
    step=5.0,
)

ring_R = st.sidebar.number_input(
    "Ring Radius R [μm] (0 = Straight)",
    min_value=0.0,
    value=100.0,
    step=10.0,
)

bottom_oxide = st.sidebar.number_input(
    "Bottom Oxide Height [μm]",
    min_value=0.01,
    value=4.0,
    step=0.5,
)

top_oxide = st.sidebar.number_input(
    "Top Oxide Height [μm]",
    min_value=0.01,
    value=1.0,
    step=0.1,
)

st.sidebar.header("🎯 Loss / Q Settings")

loss_1 = st.sidebar.number_input(
    "Loss 1 [dB/cm]",
    min_value=0.001,
    value=0.5,
    step=0.1,
)

loss_2 = st.sidebar.number_input(
    "Loss 2 [dB/cm]",
    min_value=0.001,
    value=1.5,
    step=0.1,
)

loss_3 = st.sidebar.number_input(
    "Loss 3 [dB/cm]",
    min_value=0.001,
    value=5.0,
    step=0.5,
)

custom_losses = [loss_1, loss_2, loss_3]

st.sidebar.header("🔬 Simulation Settings")

scan_mode = st.sidebar.selectbox(
    "Sweep Variable",
    ["Wavelength", "Gap"],
)

if scan_mode == "Gap":
    gap_start = st.sidebar.number_input(
        "Gap Start [μm]",
        min_value=0.001,
        value=0.2,
        step=0.05,
    )

    gap_end = st.sidebar.number_input(
        "Gap End [μm]",
        min_value=0.002,
        value=1.0,
        step=0.05,
    )

    n_gap = st.sidebar.slider(
        "Gap Points",
        min_value=3,
        max_value=51,
        value=11,
        step=1,
    )

    wavelength_ref = st.sidebar.number_input(
        "Reference Wavelength [μm]",
        min_value=0.1,
        value=1.55,
        step=0.01,
    )

else:
    lambda_start = st.sidebar.number_input(
        "Start Wavelength [μm]",
        min_value=0.1,
        value=1.5,
        step=0.05,
    )

    lambda_end = st.sidebar.number_input(
        "End Wavelength [μm]",
        min_value=0.1,
        value=1.6,
        step=0.05,
    )

    n_lambda = st.sidebar.slider(
        "Wavelength Points",
        min_value=3,
        max_value=51,
        value=11,
        step=2,
    )

polarization = st.sidebar.selectbox(
    "Polarization",
    ["ex", "ey"],
)

res_mode = st.sidebar.selectbox(
    "Mesh Resolution",
    [
        "lr (0.02μm)",
        "mr (0.01μm)",
        "hr (0.005μm)",
    ],
)

run_button = st.sidebar.button(
    "🚀 Run Simulation",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------------------
# Simulation execution
# ---------------------------------------------------------------------

if run_button:
    try:
        with st.spinner("Running simulation..."):

            if scan_mode == "Gap":
                results = run_gap_sweep(
                    w_single=w_single,
                    h_core=h_core,
                    gap_start=gap_start,
                    gap_end=gap_end,
                    n_gap=n_gap,
                    coupler_L=coupler_L,
                    ring_R=ring_R,
                    wavelength_ref=wavelength_ref,
                    polarization=polarization,
                    res_mode=res_mode,
                    top_oxide=top_oxide,
                    bottom_oxide=bottom_oxide,
                )

            else:
                if lambda_end <= lambda_start:
                    raise ValueError(
                        "End Wavelength must be greater than "
                        "Start Wavelength."
                    )

                results = run_simulation(
                    w_single=w_single,
                    h_core=h_core,
                    gap=gap,
                    coupler_L=coupler_L,
                    ring_R=ring_R,
                    lambda_start=lambda_start,
                    lambda_end=lambda_end,
                    n_lambda=n_lambda,
                    polarization=polarization,
                    res_mode=res_mode,
                    top_oxide=top_oxide,
                    bottom_oxide=bottom_oxide,
                )

                results = calculate_wavelength_results(
                    results,
                    custom_losses,
                )

            st.session_state["sim_results"] = results
            st.session_state["scan_mode"] = scan_mode

    except Exception as error:
        st.error("Simulation failed.")
        st.exception(error)
        st.stop()


# ---------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------

if "sim_results" not in st.session_state:
    st.info(
        "Set the parameters in the sidebar and click "
        "**Run Simulation**."
    )
    st.stop()


results = st.session_state["sim_results"]
saved_scan_mode = st.session_state.get("scan_mode", scan_mode)

if saved_scan_mode == "Gap":
    display_gap_results(results)
else:
    display_wavelength_results(results)
