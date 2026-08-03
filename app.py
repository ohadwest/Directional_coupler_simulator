import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
from coupler_engine import run_simulation

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

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
lambda_start = st.sidebar.number_input("Start Wavelength [μm]", value=1.5, step=0.05)
lambda_end = st.sidebar.number_input("End Wavelength [μm]", value=1.6, step=0.05)
n_lambda = st.sidebar.slider("Wavelength Points", min_value=3, max_value=21, value=11, step=2)
polarization = st.sidebar.selectbox("Polarization", options=["ex", "ey"], index=0)
res_mode = st.sidebar.selectbox("Mesh Resolution", options=["lr (0.02μm)", "mr (0.01μm)", "hr (0.005μm)"], index=0)

run_btn = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def generate_pdf_report(d, fig_disp, fig_power):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#1E3A8A'), spaceAfter=12)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#1E3A8A'), spaceBefore=10, spaceAfter=6)
    normal_style = styles['Normal']
    
    elements = []
    
    elements.append(Paragraph("Silicon Nitride Directional Coupler - Simulation Report", title_style))
    elements.append(Paragraph("Summary of parameters, modal dispersion, and optical coupling characteristics.", normal_style))
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("1. Device Parameters", heading_style))
    param_data = [
        ["Parameter", "Value", "Parameter", "Value"],
        ["Waveguide Width (w)", f"{d['w_single']} um", "Ring Radius (R)", f"{d['ring_R']} um"],
        ["Core Height (h)", f"{d['h_core']} um", "Bottom Oxide", f"{d['bottom_oxide']} um"],
        ["Gap", f"{d['gap']} um", "Top Oxide", f"{d['top_oxide']} um"],
        ["Coupler Length (L)", f"{d['coupler_L']} um", "Polarization", f"{d['polarization'].upper()}"]
    ]
    t_param = Table(param_data, colWidths=[130, 110, 130, 110])
    t_param.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(t_param)
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("2. Key Simulation Results (at Central Wavelength)", heading_style))
    res_data = [
        ["Metric", "Value"],
        ["Central Wavelength", f"{d['lambda_center_val']:.3f} um"],
        ["Coupling Coefficient (kappa)", f"{d['kappa_vec'][d['idx_center']]:.4f} um^-1"],
        ["Residual Length (L_res)", f"{d['l_residual_vec'][d['idx_center']]:.2f} um"],
        ["Power Transferred (P_cross)", f"{d['p_cross_vec'][d['idx_center']]:.1f} %"],
        [f"Loaded Q (Q_L at {d['alpha_db_vals'][1]} dB/cm)", f"{d['QL_vals'][1]/1e3:.1f} k"]
    ]
    t_res = Table(res_data, colWidths=[240, 240])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(t_res)
    elements.append(Spacer(1, 14))
    
    elements.append(Paragraph("3. Graphical Results", heading_style))
    
    img_disp_data = fig_to_bytes(fig_disp)
    img_power_data = fig_to_bytes(fig_power)
    
    img1 = RLImage(io.BytesIO(img_disp_data), width=230, height=150)
    img2 = RLImage(io.BytesIO(img_power_data), width=230, height=150)
    
    t_imgs = Table([[img1, img2]], colWidths=[240, 240])
    t_imgs.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(t_imgs)
    
    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# --- EXECUTION & DISPLAY ---
if run_btn or 'sim_results' in st.session_state:
    if run_btn:
        with st.spinner("Calculating modes and optical coupling... Please wait."):
            results = run_simulation(
                w_single, h_core, gap, coupler_L, ring_R,
                lambda_start, lambda_end, n_lambda, polarization, res_mode, top_oxide, bottom_oxide
            )
            
            alpha_db_vals = np.array(custom_losses)
            alpha_cm = alpha_db_vals * (np.log(10) / 10.0)
            L_ring_cm = results['L_ring_um'] * 1e-4
            round_trip_loss_pct = (1.0 - np.exp(-alpha_cm * L_ring_cm)) * 100.0
            
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

    d = st.session_state['sim_results']

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Central Coupling (κ)", f"{d['kappa_vec'][d['idx_center']]:.4f} μm⁻¹")
    m2.metric("Residual Length (L_res)", f"{d['l_residual_vec'][d['idx_center']]:.2f} μm")
    m3.metric("Cross Power Transferred", f"{d['p_cross_vec'][d['idx_center']]:.1f} %")
    m4.metric(f"Q_L (at α = {d['alpha_db_vals'][1]} dB/cm)", f"{d['QL_vals'][1]/1e3:.1f} k")

    st.markdown("---")
    st.subheader("📥 Export Data & Summary")
    
    df_results = pd.DataFrame({
        "Wavelength_um": d['lambda_vec'],
        "Neff_Even": d['neff_even'],
        "Neff_Odd": d['neff_odd'],
        "Kappa_1per_um": d['kappa_vec'],
        "L_residual_um": d['l_residual_vec'],
        "L_total_um": d['l_total_vec'],
        "P_cross_percent": d['p_cross_vec'],
        "P_bar_percent": d['p_bar_vec']
    })
    
    # Pre-generate figures for PDF export
    fig_disp_pdf, ax_disp_pdf = plt.subplots(figsize=(5, 3.5))
    ax_disp_pdf.plot(d['lambda_vec'], d['neff_even'], 'bo-', lw=1.5, label='n_eff Even')
    ax_disp_pdf.plot(d['lambda_vec'], d['neff_odd'], 'r^-', lw=1.5, label='n_eff Odd')
    ax_disp_pdf.grid(True)
    ax_disp_pdf.set_xlabel('Wavelength [um]')
    ax_disp_pdf.set_ylabel('n_eff')
    ax_disp_pdf.set_title("Modal Dispersion", fontsize=10)

    fig_power_pdf, ax_power_pdf = plt.subplots(figsize=(5, 3.5))
    ax_power_pdf.plot(d['lambda_vec'], d['p_cross_vec'], 'ro-', lw=1.5, label='Cross')
    ax_power_pdf.plot(d['lambda_vec'], d['p_bar_vec'], 'bo-', lw=1.5, label='Bar')
    ax_power_pdf.grid(True)
    ax_power_pdf.set_xlabel('Wavelength [um]')
    ax_power_pdf.set_ylabel('Power [%]')
    ax_power_pdf.set_title("Power Transfer", fontsize=10)

    pdf_bytes = generate_pdf_report(d, fig_disp_pdf, fig_power_pdf)
    plt.close(fig_disp_pdf)
    plt.close(fig_power_pdf)

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
            label="📕 Download Summary Report (PDF)",
            data=pdf_bytes,
            file_name="coupler_summary_report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Cross-Sections & Modes", "📈 Dispersion & Coupling", "⚡ Power Transfer", "🎯 Loss & Critical Q_L"])

    def draw_boxes(ax):
        for l, r in [(d['box1_l'], d['box1_r']), (d['box2_l'], d['box2_r'])]:
            ax.plot([l, r, r, l, l], [d['b_y'], d['b_y'], d['t_y'], d['t_y'], d['b_y']], 'k--', lw=1.5)

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            im1 = ax1.imshow(np.sqrt(d['eps_center']).T, origin='lower', extent=[d['xc'][0], d['xc'][-1], d['yc'][0], d['yc'][-1]], cmap='viridis', aspect='auto')
            fig1.colorbar(im1, ax=ax1, label='Index (n)')
            draw_boxes(ax1)
            ax1.set_title(f"Refractive Index Profile (λ = {d['lambda_center_val']:.3f} μm)")
            st.pyplot(fig1)
            st.download_button("💾 Save Index Profile PNG", data=fig_to_bytes(fig1), file_name="index_profile.png", mime="image/png")

            fig3, ax3 = plt.subplots(figsize=(6, 4))
            im3 = ax3.imshow(d['phi_odd'].T, origin='lower', extent=[d['xc'][0], d['xc'][-1], d['yc'][0], d['yc'][-1]], cmap='jet', vmin=-1, vmax=1, aspect='auto')
            fig3.colorbar(im3, ax=ax3, label='Field')
            draw_boxes(ax3)
            ax3.set_title(f"Antisymmetric (Odd) Mode ({d['polarization'].upper()})")
            st.pyplot(fig3)
            st.download_button("💾 Save Odd Mode PNG", data=fig_to_bytes(fig3), file_name="odd_mode.png", mime="image/png")

        with col_b:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            im2 = ax2.imshow(d['phi_even'].T, origin='lower', extent=[d['xc'][0], d['xc'][-1], d['yc'][0], d['yc'][-1]], cmap='jet', vmin=0, vmax=1, aspect='auto')
            fig2.colorbar(im2, ax=ax2, label='Field')
            draw_boxes(ax2)
            ax2.set_title(f"Symmetric (Even) Mode ({d['polarization'].upper()})")
            st.pyplot(fig2)
            st.download_button("💾 Save Even Mode PNG", data=fig_to_bytes(fig2), file_name="even_mode.png", mime="image/png")

            fig4, ax4 = plt.subplots(figsize=(6, 4))
            ax4.plot(d['xc'], d['phi_even'][:, d['mid_y_idx']], 'b-', lw=2, label='Even')
            ax4.plot(d['xc'], d['phi_odd'][:, d['mid_y_idx']], 'r--', lw=2, label='Odd')
            ax4.grid(True)
            ax4.legend()
            ax4.set_title("1D Field Profiles at Core Center")
            st.pyplot(fig4)
            st.download_button("💾 Save 1D Profile PNG", data=fig_to_bytes(fig4), file_name="1d_profiles.png", mime="image/png")

    with tab2:
        col_c, col_d = st.columns(2)
        with col_c:
            fig5, ax5 = plt.subplots(figsize=(6, 4))
            ax5.plot(d['lambda_vec'], d['neff_even'], 'bo-', lw=2, label='n_eff Even')
            ax5.plot(d['lambda_vec'], d['neff_odd'], 'r^-', lw=2, label='n_eff Odd')
            ax5.grid(True)
            ax5.legend()
            ax5.set_xlabel('Wavelength [μm]')
            ax5.set_ylabel('Effective Index (n_eff)')
            ax5.set_title("Supermode Dispersion Curves")
            st.pyplot(fig5)
            st.download_button("💾 Save Dispersion Graph PNG", data=fig_to_bytes(fig5), file_name="dispersion.png", mime="image/png")

        with col_d:
            fig6, ax6_left = plt.subplots(figsize=(6, 4))
            ax6_right = ax6_left.twinx()
            ax6_left.plot(d['lambda_vec'], d['kappa_vec'], 'kd-', lw=2, label='Kappa')
            ax6_right.plot(d['lambda_vec'], d['l_residual_vec'], 'ms-', lw=2, label='L_residual')
            ax6_left.grid(True)
            ax6_left.set_xlabel('Wavelength [μm]')
            ax6_left.set_ylabel('κ [μm⁻¹]', color='k')
            ax6_right.set_ylabel('L_residual [μm]', color='m')
            ax6_left.set_title("Coupling Coefficient κ & Residual Length")
            st.pyplot(fig6)
            st.download_button("💾 Save Kappa Graph PNG", data=fig_to_bytes(fig6), file_name="kappa_coupling.png", mime="image/png")

    with tab3:
        fig7, ax7 = plt.subplots(figsize=(9, 4.5))
        ax7.plot(d['lambda_vec'], d['p_cross_vec'], 'ro-', lw=2, label='Cross Port Power (Transferred)')
        ax7.plot(d['lambda_vec'], d['p_bar_vec'], 'bo-', lw=2, label='Bar Port Power (Remaining)')
        ax7.grid(True)
        ax7.set_ylim(0, 105)
        ax7.set_xlabel('Wavelength [μm]')
        ax7.set_ylabel('Power Transfer [%]')
        ax7.legend()
        ax7.set_title("Power Transfer Ratio vs. Wavelength")
        st.pyplot(fig7)
        st.download_button("💾 Save Power Transfer PNG", data=fig_to_bytes(fig7), file_name="power_transfer.png", mime="image/png")

    with tab4:
        fig8, ax8 = plt.subplots(figsize=(9, 4.5))
        ax8.plot(d['lambda_vec'], d['p_cross_vec'], 'ro-', lw=2.5, label='Coupled Power P_cross')
        colors = ['g--', 'm--', 'k--']
        for k in range(3):
            loss_v = d['round_trip_loss_pct'][k]
            ql_v = d['QL_vals'][k] / 1e3
            alpha_db = d['alpha_db_vals'][k]
            label_text = f"Loss = {loss_v:.3f}% (α={alpha_db}dB/cm, QL≈{ql_v:.1f}k)"
            ax8.axhline(loss_v, color=colors[k][0], linestyle='--', lw=1.8, label=label_text)
        ax8.grid(True)
        ax8.set_xlabel('Wavelength [μm]')
        ax8.set_ylabel('Power [%]')
        ax8.legend(fontsize=9)
        ax8.set_title(f"Ring Coupling vs. Loss & Critical Q_L (L_ring = {d['L_ring_um']:.1f} μm)")
        st.pyplot(fig8)
        st.download_button("💾 Save Ring Loss PNG", data=fig_to_bytes(fig8), file_name="ring_loss_QL.png", mime="image/png")

else:
    st.info("👈 Set your parameters in the sidebar and click **Run Simulation** to view the browser dashboard!")
