# Silicon Nitride Directional & Ring Coupler Solver
# Version: 1.5.0
# Written: 2026-09-07 09:57:50
# Recent changes:
# - Added selectable sweep mode: Wavelength or Gap.
# - Added fixed-reference-wavelength gap sweeps using the existing solver.
# - Updated dispersion, coupling, power, loss, and Q-related plots to use
#   the selected sweep parameter on the horizontal axis.
# - Updated CSV export to label the scanned variable correctly.
# - Preserved the existing dashboard layout, tabs, mode figures, PNG exports,
#   comprehensive PDF report, and wavelength-sweep behavior.
# - Added live gap-sweep progress, percentage complete, elapsed time, and
#   remaining-time estimation based on the first completed simulation.
# - Added visible version and build metadata to the Streamlit sidebar.
# - Added live progress and ETA for wavelength sweeps.
# - Fixed invalid sparse-matrix wraparound connections in the mode solver.

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import time
from coupler_engine import run_simulation

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

APP_VERSION = "1.5.0"
APP_BUILD_DATE = "2026-09-07 09:57:50"

st.set_page_config(
    page_title="Silicon Photonics Coupler Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Silicon Nitride Directional & Ring Coupler Solver")
st.markdown("### 2D Semi-Vectorial Finite Difference Mode Solver & Coupled Mode Analysis")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🛠️ Coupler Parameters")
w_single = st.sidebar.number_input("Waveguide Width w [μm]", value=1.0, step=0.1)
h_core = st.sidebar.number_input("Waveguide Height h [μm]", value=0.3, step=0.05)
gap = st.sidebar.number_input("Coupler Gap [μm]", value=0.3, step=0.05)
coupler_L = st.sidebar.number_input("Straight Length L [μm]", value=35.0, step=5.0)
ring_R = st.sidebar.number_input("Ring Radius R [μm] (0=Straight)", value=100.0, step=10.0)
bottom_oxide = st.sidebar.number_input("Bottom Oxide Height [μm]", value=4.0, step=0.5)
top_oxide = st.sidebar.number_input("Top Oxide Height [μm]", value=1.0, step=0.1)

st.sidebar.header("🎯 Loss / Q_L Settings")
st.sidebar.markdown("Set 3 loss values [dB/cm] for critical coupling analysis:")
loss_1 = st.sidebar.number_input("Loss 1 [dB/cm]", value=0.5, step=0.1)
loss_2 = st.sidebar.number_input("Loss 2 [dB/cm]", value=1.5, step=0.1)
loss_3 = st.sidebar.number_input("Loss 3 [dB/cm]", value=5.0, step=0.5)
custom_losses = [loss_1, loss_2, loss_3]

st.sidebar.header("🔬 Simulation Settings")
scan_mode = st.sidebar.selectbox(
    "Scan Parameter",
    options=["Wavelength", "Gap"],
    index=0,
)

if scan_mode == "Wavelength":
    lambda_start = st.sidebar.number_input("Start Wavelength [μm]", value=1.5, step=0.05)
    lambda_end = st.sidebar.number_input("End Wavelength [μm]", value=1.6, step=0.05)
    n_lambda = st.sidebar.slider("Wavelength Points", min_value=3, max_value=21, value=11, step=2)
else:
    gap_start = st.sidebar.number_input("Start Gap [μm]", min_value=0.001, value=0.2, step=0.05)
    gap_end = st.sidebar.number_input("End Gap [μm]", min_value=0.002, value=1.0, step=0.05)
    n_gap = st.sidebar.slider("Gap Points", min_value=3, max_value=51, value=11, step=1)
    reference_wavelength = st.sidebar.number_input("Reference Wavelength [μm]", min_value=0.1, value=1.55, step=0.01)

polarization_label = st.sidebar.selectbox(
    "Polarization",
    options=["TE-like (Ex)", "TM-like (Ey)"],
    index=0,
)
polarization = {
    "TE-like (Ex)": "ex",
    "TM-like (Ey)": "ey",
}[polarization_label]
res_mode = st.sidebar.selectbox("Mesh Resolution", options=["lr (0.02μm)", "mr (0.01μm)", "hr (0.005μm)"], index=0)

run_btn = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

st.sidebar.divider()
st.sidebar.caption("Dashboard build")
st.sidebar.code(
    f"Version: {APP_VERSION}\n"
    f"Built:   {APP_BUILD_DATE}",
    language="text",
)

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def generate_pdf_report(d, fig_dict):
    """Generates a comprehensive PDF report containing all parameters, tables, and figures."""
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#1E3A8A'), spaceAfter=8)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#1E3A8A'), spaceBefore=8, spaceAfter=4)
    normal_style = styles['Normal']
    
    elements = []
    
    # --- HEADER & PARAMETERS ---
    elements.append(Paragraph("Silicon Nitride Directional Coupler - Comprehensive Report", title_style))
    elements.append(Paragraph("Detailed analysis including geometry, modal profiles, dispersion, power transfer, and Q-factor calculations.", normal_style))
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph("1. Simulation Parameters", heading_style))
    param_data = [
        ["Parameter", "Value", "Parameter", "Value"],
        ["Waveguide Width (w)", f"{d['w_single']} um", "Ring Radius (R)", f"{d['ring_R']} um"],
        ["Core Height (h)", f"{d['h_core']} um", "Bottom Oxide", f"{d['bottom_oxide']} um"],
        ["Gap", f"{d['gap']} um", "Top Oxide", f"{d['top_oxide']} um"],
        ["Coupler Length (L)", f"{d['coupler_L']} um", "Polarization", f"{d['polarization'].upper()}"],
        ["Start Wavelength", f"{d['lambda_vec'][0]:.3f} um", "End Wavelength", f"{d['lambda_vec'][-1]:.3f} um"]
    ]
    t_param = Table(param_data, colWidths=[130, 110, 130, 110])
    t_param.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(t_param)
    elements.append(Spacer(1, 8))
    
    elements.append(Paragraph("2. Key Results (Central Wavelength)", heading_style))
    res_data = [
        ["Metric", "Value"],
        ["Central Wavelength", f"{d['lambda_center_val']:.3f} um"],
        ["Coupling Coefficient (kappa)", f"{d['kappa_vec'][d['idx_center']]:.4f} um^-1"],
        ["Residual Length (L_res)", f"{d['l_residual_vec'][d['idx_center']]:.2f} um"],
        ["Power Transferred (P_cross)", f"{d['p_cross_vec'][d['idx_center']]:.1f} %"],
        [f"Loaded Q (Q_L at {d['alpha_db_vals'][0]} dB/cm)", f"{d['QL_vals'][0]/1e3:.1f} k"],
        [f"Loaded Q (Q_L at {d['alpha_db_vals'][1]} dB/cm)", f"{d['QL_vals'][1]/1e3:.1f} k"],
        [f"Loaded Q (Q_L at {d['alpha_db_vals'][2]} dB/cm)", f"{d['QL_vals'][2]/1e3:.1f} k"]
    ]
    t_res = Table(res_data, colWidths=[240, 240])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(t_res)
    elements.append(Spacer(1, 10))
    
    # --- PAGE 1 FIGURES: MODES & CROSS SECTIONS ---
    elements.append(Paragraph("3. Cross-Sections & Mode Profiles", heading_style))
    
    img_index = RLImage(io.BytesIO(fig_to_bytes(fig_dict['index'])), width=235, height=155)
    img_even = RLImage(io.BytesIO(fig_to_bytes(fig_dict['even'])), width=235, height=155)
    img_odd = RLImage(io.BytesIO(fig_to_bytes(fig_dict['odd'])), width=235, height=155)
    img_1d = RLImage(io.BytesIO(fig_to_bytes(fig_dict['1d'])), width=235, height=155)
    
    t_modes = Table([
        [img_index, img_even],
        [img_odd, img_1d]
    ], colWidths=[240, 240])
    t_modes.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 2),
    ]))
    elements.append(t_modes)
    
    # --- PAGE 2: DISPERSION, POWER & LOSS ---
    elements.append(PageBreak())
    elements.append(Paragraph("4. Dispersion & Optical Coupling Curves", heading_style))
    
    img_disp = RLImage(io.BytesIO(fig_to_bytes(fig_dict['disp'])), width=235, height=155)
    img_kappa = RLImage(io.BytesIO(fig_to_bytes(fig_dict['kappa'])), width=235, height=155)
    
    t_disp = Table([[img_disp, img_kappa]], colWidths=[240, 240])
    t_disp.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 2),
    ]))
    elements.append(t_disp)
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph("5. Power Transfer & Ring Coupling Analysis", heading_style))
    
    img_power = RLImage(io.BytesIO(fig_to_bytes(fig_dict['power'])), width=235, height=155)
    img_loss = RLImage(io.BytesIO(fig_to_bytes(fig_dict['loss'])), width=235, height=155)
    
    t_power = Table([[img_power, img_loss]], colWidths=[240, 240])
    t_power.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 2),
    ]))
    elements.append(t_power)
    
    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def run_gap_sweep(
    w_single, h_core, gap_start, gap_end, n_gap, coupler_L, ring_R,
    reference_wavelength, polarization, res_mode, top_oxide, bottom_oxide
):
    """Run the existing solver repeatedly at a fixed wavelength while varying gap."""
    if gap_end <= gap_start:
        raise ValueError("End Gap must be greater than Start Gap.")

    gap_vec = np.linspace(gap_start, gap_end, n_gap)
    delta_lambda = 1e-6
    sweep_results = []
    total_simulations = len(gap_vec)
    progress_bar = st.progress(0, text=f"Simulation 0/{total_simulations} (0.0%)")
    progress_status = st.empty()
    sweep_start_time = time.perf_counter()
    first_simulation_time = None

    for simulation_number, current_gap in enumerate(gap_vec, start=1):
        completed_simulations = simulation_number - 1
        completed_percentage = 100.0 * completed_simulations / total_simulations
        progress_bar.progress(
            completed_simulations / total_simulations,
            text=f"Running simulation {simulation_number}/{total_simulations} "
                 f"({completed_percentage:.1f}% complete)"
        )
        progress_status.info(
            f"Gap sweep: running simulation {simulation_number}/{total_simulations} "
            f"({completed_percentage:.1f}% complete)"
        )

        def update_solver_progress(completed_wavelengths, total_wavelengths):
            fraction = completed_wavelengths / total_wavelengths
            overall_fraction = (
                (simulation_number - 1) + fraction
            ) / total_simulations
            overall_percentage = 100.0 * overall_fraction
            progress_bar.progress(
                overall_fraction,
                text=(
                    f"Simulation {simulation_number}/{total_simulations} "
                    f"({overall_percentage:.1f}% complete)"
                )
            )
            progress_status.info(
                f"Gap sweep: simulation {simulation_number}/{total_simulations} "
                f"in progress ({overall_percentage:.1f}% complete)"
            )

        sweep_results.append(run_simulation(
            w_single, h_core, float(current_gap), coupler_L, ring_R,
            reference_wavelength - delta_lambda,
            reference_wavelength + delta_lambda,
            3, polarization, res_mode, top_oxide, bottom_oxide,
            progress_callback=update_solver_progress
        ))

        elapsed_time = time.perf_counter() - sweep_start_time
        if first_simulation_time is None:
            first_simulation_time = elapsed_time

        # Use the first completed simulation for the initial ETA. Once more
        # samples are available, use the measured average time per simulation.
        if simulation_number == 1:
            estimated_total_time = first_simulation_time * total_simulations
        else:
            estimated_total_time = (elapsed_time / simulation_number) * total_simulations

        remaining_time = max(0.0, estimated_total_time - elapsed_time)
        percentage = 100.0 * simulation_number / total_simulations
        progress_bar.progress(
            simulation_number / total_simulations,
            text=f"Simulation {simulation_number}/{total_simulations} ({percentage:.1f}%)"
        )
        progress_status.info(
            f"Gap sweep: {simulation_number}/{total_simulations} simulations completed "
            f"({percentage:.1f}%) · elapsed {elapsed_time:.1f} s · "
            f"estimated remaining {remaining_time:.1f} s"
        )

    progress_bar.progress(1.0, text=f"Simulation {total_simulations}/{total_simulations} (100.0%)")
    progress_status.success(
        f"Gap sweep completed: {total_simulations}/{total_simulations} simulations "
        f"in {time.perf_counter() - sweep_start_time:.1f} s"
    )

    center_index = len(gap_vec) // 2
    output = sweep_results[center_index].copy()
    wavelength_index = 1

    for key in (
        "neff_even", "neff_odd", "kappa_vec", "l_residual_vec",
        "l_total_vec", "p_cross_vec", "p_bar_vec"
    ):
        output[key] = np.asarray([
            result[key][wavelength_index] for result in sweep_results
        ])

    output["gap_vec"] = np.asarray(gap_vec)
    output["scan_mode"] = "Gap"
    output["scan_vec"] = output["gap_vec"]
    output["scan_label"] = "Gap [μm]"
    output["reference_wavelength"] = reference_wavelength
    output["lambda_center_val"] = reference_wavelength
    output["idx_center"] = center_index
    output["gap"] = float(gap_vec[center_index])

    # Keep the narrow wavelength vector from the reference calculation for
    # the existing report metadata and compute Q at the reference wavelength.
    q_result = sweep_results[center_index]
    q_neff_avg = (np.asarray(q_result["neff_even"]) + np.asarray(q_result["neff_odd"])) / 2.0
    q_lambda = np.asarray(q_result["lambda_vec"])
    if len(q_lambda) > 1 and q_lambda[-1] != q_lambda[0]:
        dneff_dlambda = (q_neff_avg[-1] - q_neff_avg[0]) / ((q_lambda[-1] - q_lambda[0]) * 1e-4)
        n_group = q_neff_avg[1] - (reference_wavelength * 1e-4) * dneff_dlambda
    else:
        n_group = q_neff_avg[0]

    output["q_n_group"] = n_group
    return output

# --- EXECUTION & DISPLAY ---
if run_btn or 'sim_results' in st.session_state:
    if run_btn:
        with st.spinner("Calculating modes and optical coupling... Please wait."):
            if scan_mode == "Gap":
                results = run_gap_sweep(
                    w_single, h_core, gap_start, gap_end, n_gap,
                    coupler_L, ring_R, reference_wavelength,
                    polarization, res_mode, top_oxide, bottom_oxide
                )
            else:
                wavelength_start_time = time.perf_counter()
                wavelength_progress = st.progress(
                    0,
                    text=f"Wavelength simulation 0/{n_lambda} (0.0%)"
                )
                wavelength_status = st.empty()

                def update_wavelength_progress(completed_points, total_points):
                    elapsed_time = time.perf_counter() - wavelength_start_time
                    percentage = 100.0 * completed_points / total_points
                    if completed_points > 0:
                        estimated_total_time = (
                            elapsed_time / completed_points
                        ) * total_points
                        remaining_time = max(
                            0.0,
                            estimated_total_time - elapsed_time,
                        )
                    else:
                        remaining_time = 0.0

                    wavelength_progress.progress(
                        completed_points / total_points,
                        text=(
                            f"Wavelength simulation "
                            f"{completed_points}/{total_points} "
                            f"({percentage:.1f}%)"
                        ),
                    )
                    wavelength_status.info(
                        f"Wavelength sweep: {completed_points}/{total_points} "
                        f"points completed ({percentage:.1f}%) · "
                        f"elapsed {elapsed_time:.1f} s · "
                        f"estimated remaining {remaining_time:.1f} s"
                    )

                results = run_simulation(
                    w_single, h_core, gap, coupler_L, ring_R,
                    lambda_start, lambda_end, n_lambda,
                    polarization, res_mode, top_oxide, bottom_oxide,
                    progress_callback=update_wavelength_progress
                )
                wavelength_progress.progress(
                    1.0,
                    text=f"Wavelength simulation {n_lambda}/{n_lambda} (100.0%)"
                )
                wavelength_status.success(
                    f"Wavelength sweep completed: {n_lambda}/{n_lambda} points"
                )
                results["scan_mode"] = "Wavelength"
                results["scan_vec"] = results["lambda_vec"]
                results["scan_label"] = "Wavelength [μm]"
            
            alpha_db_vals = np.array(custom_losses)
            alpha_cm = alpha_db_vals * (np.log(10) / 10.0)
            L_ring_cm = results['L_ring_um'] * 1e-4
            round_trip_loss_pct = (1.0 - np.exp(-alpha_cm * L_ring_cm)) * 100.0
            
            if scan_mode == "Gap":
                n_group = results["q_n_group"]
                lambda_cm_center = results["lambda_center_val"] * 1e-4
            else:
                neff_avg_vec = (results['neff_even'] + results['neff_odd']) / 2.0
                lambda_cm_center = results['lambda_center_val'] * 1e-4
                dneff_dlambda = (neff_avg_vec[-1] - neff_avg_vec[0]) / ((results['lambda_vec'][-1] - results['lambda_vec'][0]) * 1e-4)
                n_group = neff_avg_vec[results['idx_center']] - lambda_cm_center * dneff_dlambda
            
            Q0_vals = (2.0 * np.pi * n_group) / (lambda_cm_center * alpha_cm)
            QL_vals = Q0_vals / 2.0
            
            results['alpha_db_vals'] = alpha_db_vals
            results['round_trip_loss_pct'] = round_trip_loss_pct
            results['QL_vals'] = QL_vals
            
            st.session_state['sim_results'] = results
            st.session_state['scan_mode'] = scan_mode

    d = st.session_state['sim_results']

    active_scan_mode = d.get('scan_mode', st.session_state.get('scan_mode', 'Wavelength'))
    x_values = d['gap_vec'] if active_scan_mode == 'Gap' else d['lambda_vec']
    x_label = 'Gap [μm]' if active_scan_mode == 'Gap' else 'Wavelength [μm]'
    x_title = 'Gap' if active_scan_mode == 'Gap' else 'Wavelength'

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Central Coupling (κ)", f"{d['kappa_vec'][d['idx_center']]:.4f} μm⁻¹")
    m2.metric("Residual Length (L_res)", f"{d['l_residual_vec'][d['idx_center']]:.2f} μm")
    m3.metric("Cross Power Transferred", f"{d['p_cross_vec'][d['idx_center']]:.1f} %")
    m4.metric(f"Q_L (at α = {d['alpha_db_vals'][1]} dB/cm)", f"{d['QL_vals'][1]/1e3:.1f} k")

    st.markdown("---")
    st.subheader("📥 Export Data & Full Report")
    
    df_results = pd.DataFrame({
        ("Gap_um" if active_scan_mode == 'Gap' else "Wavelength_um"): x_values,
        "Neff_Even": d['neff_even'],
        "Neff_Odd": d['neff_odd'],
        "Kappa_1per_um": d['kappa_vec'],
        "L_residual_um": d['l_residual_vec'],
        "L_total_um": d['l_total_vec'],
        "P_cross_percent": d['p_cross_vec'],
        "P_bar_percent": d['p_bar_vec']
    })
    
    # --- GENERATE FIGURES FOR ALL TABS ---
    def draw_boxes(ax):
        for l, r in [(d['box1_l'], d['box1_r']), (d['box2_l'], d['box2_r'])]:
            ax.plot([l, r, r, l, l], [d['b_y'], d['b_y'], d['t_y'], d['t_y'], d['b_y']], 'k--', lw=1.5)

    # 1. Index Profile
    fig_idx, ax_idx = plt.subplots(figsize=(6, 4))
    im_idx = ax_idx.imshow(np.sqrt(d['eps_center']).T, origin='lower', extent=[d['xc'][0], d['xc'][-1], d['yc'][0], d['yc'][-1]], cmap='viridis', aspect='auto')
    fig_idx.colorbar(im_idx, ax=ax_idx, label='Index (n)')
    draw_boxes(ax_idx)
    ax_idx.set_title(f"Refractive Index Profile (λ = {d['lambda_center_val']:.3f} μm)")

    # 2. Even Mode
    fig_even, ax_even = plt.subplots(figsize=(6, 4))
    im_even = ax_even.imshow(d['phi_even'].T, origin='lower', extent=[d['xc'][0], d['xc'][-1], d['yc'][0], d['yc'][-1]], cmap='jet', vmin=0, vmax=1, aspect='auto')
    fig_even.colorbar(im_even, ax=ax_even, label='Field')
    draw_boxes(ax_even)
    ax_even.set_title(f"Symmetric (Even) Mode ({d['polarization'].upper()})")

    # 3. Odd Mode
    fig_odd, ax_odd = plt.subplots(figsize=(6, 4))
    im_odd = ax_odd.imshow(d['phi_odd'].T, origin='lower', extent=[d['xc'][0], d['xc'][-1], d['yc'][0], d['yc'][-1]], cmap='jet', vmin=-1, vmax=1, aspect='auto')
    fig_odd.colorbar(im_odd, ax=ax_odd, label='Field')
    draw_boxes(ax_odd)
    ax_odd.set_title(f"Antisymmetric (Odd) Mode ({d['polarization'].upper()})")

    # 4. 1D Profiles
    fig_1d, ax_1d = plt.subplots(figsize=(6, 4))
    ax_1d.plot(d['xc'], d['phi_even'][:, d['mid_y_idx']], 'b-', lw=2, label='Even')
    ax_1d.plot(d['xc'], d['phi_odd'][:, d['mid_y_idx']], 'r--', lw=2, label='Odd')
    ax_1d.grid(True)
    ax_1d.legend()
    ax_1d.set_title("1D Field Profiles at Core Center")

    # 5. Dispersion
    fig_disp, ax_disp = plt.subplots(figsize=(6, 4))
    ax_disp.plot(x_values, d['neff_even'], 'bo-', lw=2, label='n_eff Even')
    ax_disp.plot(x_values, d['neff_odd'], 'r^-', lw=2, label='n_eff Odd')
    ax_disp.grid(True)
    ax_disp.legend()
    ax_disp.set_xlabel(x_label)
    ax_disp.set_ylabel('Effective Index (n_eff)')
    ax_disp.set_title(f"Supermode Dispersion Curves vs. {x_title}")

    # 6. Kappa & L_res
    fig_kappa, ax_kappa_left = plt.subplots(figsize=(6, 4))
    ax_kappa_right = ax_kappa_left.twinx()
    ax_kappa_left.plot(x_values, d['kappa_vec'], 'kd-', lw=2, label='Kappa')
    ax_kappa_right.plot(x_values, d['l_residual_vec'], 'ms-', lw=2, label='L_residual')
    ax_kappa_left.grid(True)
    ax_kappa_left.set_xlabel(x_label)
    ax_kappa_left.set_ylabel('κ [μm⁻¹]', color='k')
    ax_kappa_right.set_ylabel('L_residual [μm]', color='m')
    ax_kappa_left.set_title(f"Coupling Coefficient κ & Residual Length vs. {x_title}")

    # 7. Power Transfer
    fig_power, ax_power = plt.subplots(figsize=(7, 4))
    ax_power.plot(x_values, d['p_cross_vec'], 'ro-', lw=2, label='Cross Port Power')
    ax_power.plot(x_values, d['p_bar_vec'], 'bo-', lw=2, label='Bar Port Power')
    ax_power.grid(True)
    ax_power.set_ylim(0, 105)
    ax_power.set_xlabel(x_label)
    ax_power.set_ylabel('Power Transfer [%]')
    ax_power.legend()
    ax_power.set_title(f"Power Transfer Ratio vs. {x_title}")

    # 8. Loss & Q_L
    fig_loss, ax_loss = plt.subplots(figsize=(7, 4))
    ax_loss.plot(x_values, d['p_cross_vec'], 'ro-', lw=2.5, label='Coupled Power P_cross')
    colors_list = ['g--', 'm--', 'k--']
    for k in range(3):
        loss_v = d['round_trip_loss_pct'][k]
        ql_v = d['QL_vals'][k] / 1e3
        alpha_db = d['alpha_db_vals'][k]
        label_text = f"Loss = {loss_v:.3f}% (α={alpha_db}dB/cm, QL≈{ql_v:.1f}k)"
        ax_loss.axhline(loss_v, color=colors_list[k][0], linestyle='--', lw=1.8, label=label_text)
    ax_loss.grid(True)
    ax_loss.set_xlabel(x_label)
    ax_loss.set_ylabel('Power [%]')
    ax_loss.legend(fontsize=8)
    ax_loss.set_title(f"Ring Coupling vs. Loss & Critical Q_L vs. {x_title} (L_ring = {d['L_ring_um']:.1f} μm)")

    # Dictionary of all generated figures for the PDF
    all_figs = {
        'index': fig_idx, 'even': fig_even, 'odd': fig_odd, '1d': fig_1d,
        'disp': fig_disp, 'kappa': fig_kappa, 'power': fig_power, 'loss': fig_loss
    }

    pdf_bytes = generate_pdf_report(d, all_figs)

    # --- DOWNLOAD BUTTONS ---
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        csv_bytes = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Raw Data (CSV)",
            data=csv_bytes,
            file_name="simulation_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_exp2:
        st.download_button(
            label="📕 Download Comprehensive PDF Report",
            data=pdf_bytes,
            file_name="coupler_comprehensive_report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

    st.markdown("---")

    # --- DISPLAY TABS IN STREAMLIT ---
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Cross-Sections & Modes", "📈 Dispersion & Coupling", "⚡ Power Transfer", "🎯 Loss & Critical Q_L"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.pyplot(fig_idx)
            st.download_button("💾 Save Index Profile PNG", data=fig_to_bytes(fig_idx), file_name="index_profile.png", mime="image/png")

            st.pyplot(fig_odd)
            st.download_button("💾 Save Odd Mode PNG", data=fig_to_bytes(fig_odd), file_name="odd_mode.png", mime="image/png")

        with col_b:
            st.pyplot(fig_even)
            st.download_button("💾 Save Even Mode PNG", data=fig_to_bytes(fig_even), file_name="even_mode.png", mime="image/png")

            st.pyplot(fig_1d)
            st.download_button("💾 Save 1D Profile PNG", data=fig_to_bytes(fig_1d), file_name="1d_profiles.png", mime="image/png")

    with tab2:
        col_c, col_d = st.columns(2)
        with col_c:
            st.pyplot(fig_disp)
            st.download_button("💾 Save Dispersion Graph PNG", data=fig_to_bytes(fig_disp), file_name="dispersion.png", mime="image/png")

        with col_d:
            st.pyplot(fig_kappa)
            st.download_button("💾 Save Kappa Graph PNG", data=fig_to_bytes(fig_kappa), file_name="kappa_coupling.png", mime="image/png")

    with tab3:
        st.pyplot(fig_power)
        st.download_button("💾 Save Power Transfer PNG", data=fig_to_bytes(fig_power), file_name="power_transfer.png", mime="image/png")

    with tab4:
        st.pyplot(fig_loss)
        st.download_button("💾 Save Ring Loss PNG", data=fig_to_bytes(fig_loss), file_name="ring_loss_QL.png", mime="image/png")

    # Close matplotlib figures to free memory
    for fig_obj in all_figs.values():
        plt.close(fig_obj)

else:
    st.info("👈 Set your parameters in the sidebar and click **Run Simulation** to view the browser dashboard!")
