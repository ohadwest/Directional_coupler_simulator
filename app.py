

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
scan_mode = st.sidebar.selectbox("Sweep Variable", options=["Wavelength", "Gap"], index=0)

if scan_mode == "Gap":
    gap_start = st.sidebar.number_input("Gap Start [μm]", value=0.2, step=0.05)
    gap_end = st.sidebar.number_input("Gap End [μm]", value=1.0, step=0.05)
    n_gap = st.sidebar.slider("Gap Points", min_value=3, max_value=31, value=11, step=1)
    wavelength_ref = st.sidebar.number_input("Reference Wavelength [μm]", value=1.55, step=0.01)
else:
    lambda_start = st.sidebar.number_input("Start Wavelength [μm]", value=1.5, step=0.05)
    lambda_end = st.sidebar.number_input("End Wavelength [μm]", value=1.6, step=0.05)
    n_lambda = st.sidebar.slider("Wavelength Points", min_value=3, max_value=21, value=11, step=2)

polarization = st.sidebar.selectbox("Polarization", options=["ex", "ey"], index=0)
res_mode = st.sidebar.selectbox("Mesh Resolution", options=["lr (0.02μm)", "mr (0.01μm)", "hr (0.005μm)"], index=0)

run_btn = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

# --- HELPER: GAP SWEEP ---
def run_gap_sweep(w_single, h_core, gap_start, gap_end, n_gap, coupler_L, ring_R,
                  wavelength_ref, polarization, res_mode, top_oxide, bottom_oxide):
    gap_vec = np.linspace(gap_start, gap_end, n_gap)

    out = {
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

    for g in gap_vec:
        res = run_simulation(
            w_single=w_single,
            h_core=h_core,
            gap=g,
            coupler_L=coupler_L,
            ring_R=ring_R,
            lambda_start=wavelength_ref,
            lambda_end=wavelength_ref,
            n_lambda=1,
            polarization=polarization,
            res_mode=res_mode,
            top_oxide=top_oxide,
            bottom_oxide=bottom_oxide
        )

        idx = 0
        if len(res["lambda_vec"]) > 1:
            idx = int(np.argmin(np.abs(np.asarray(res["lambda_vec"]) - wavelength_ref)))

        out["gap_vec"].append(g)
        out["kappa_vec"].append(res["kappa_vec"][idx])
        out["p_cross_vec"].append(res["p_cross_vec"][idx])
        out["p_bar_vec"].append(res["p_bar_vec"][idx])
        out["l_residual_vec"].append(res["l_residual_vec"][idx])
        out["l_total_vec"].append(res["l_total_vec"][idx])
        out["neff_even"].append(res["neff_even"][idx])
        out["neff_odd"].append(res["neff_odd"][idx])

    out["gap_vec"] = np.asarray(out["gap_vec"])
    out["kappa_vec"] = np.asarray(out["kappa_vec"])
    out["p_cross_vec"] = np.asarray(out["p_cross_vec"])
    out["p_bar_vec"] = np.asarray(out["p_bar_vec"])
    out["l_residual_vec"] = np.asarray(out["l_residual_vec"])
    out["l_total_vec"] = np.asarray(out["l_total_vec"])
    out["neff_even"] = np.asarray(out["neff_even"])
    out["neff_odd"] = np.asarray(out["neff_odd"])

    return out

# --- EXECUTION & DISPLAY ---
if run_btn or 'sim_results' in st.session_state:
    if run_btn:
        with st.spinner("Calculating modes and optical coupling... Please wait."):
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
                    bottom_oxide=bottom_oxide
                )
            else:
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
            st.session_state['scan_mode'] = scan_mode

    d = st.session_state['sim_results']
    scan_mode = st.session_state.get("scan_mode", scan_mode)

    # --- NEW BRANCH: GAP SWEEP MODE ---
    if scan_mode == "Gap":
        st.title("📈 Gap Sweep Analysis")
        st.markdown(f"### Sweeping gap from {d['gap_vec'][0]:.2f} μm to {d['gap_vec'][-1]:.2f} μm at λ = {d['lambda_ref']:.3f} μm")

        c1, c2, c3 = st.columns(3)
        c1.metric("Max κ", f"{np.max(d['kappa_vec']):.4f} μm⁻¹")
        c2.metric("Max Cross Power", f"{np.max(d['p_cross_vec']):.1f} %")
        c3.metric("Gap at Max Cross Power", f"{d['gap_vec'][np.argmax(d['p_cross_vec'])]:.2f} μm")

        fig_gap_kappa, ax_gap_kappa = plt.subplots(figsize=(8, 5))
        ax_gap_kappa.plot(d["gap_vec"], d["kappa_vec"], "o-", color="darkorange", lw=2, label="κ")
        ax_gap_kappa.set_xlabel("Gap [μm]")
        ax_gap_kappa.set_ylabel("κ [μm⁻¹]")
        ax_gap_kappa.grid(True)
        ax_gap_kappa.set_title("Coupling Coefficient vs. Gap")
        ax_gap_kappa.legend()
        st.pyplot(fig_gap_kappa)

        fig_gap_power, ax_gap_power = plt.subplots(figsize=(8, 5))
        ax_gap_power.plot(d["gap_vec"], d["p_cross_vec"], "s-", color="royalblue", lw=2, label="Cross Power")
        ax_gap_power.plot(d["gap_vec"], d["p_bar_vec"], "^-", color="forestgreen", lw=2, label="Bar Power")
        ax_gap_power.set_xlabel("Gap [μm]")
        ax_gap_power.set_ylabel("Power Transfer [%]")
        ax_gap_power.set_ylim(0, 105)
        ax_gap_power.grid(True)
        ax_gap_power.set_title("Power Transfer vs. Gap")
        ax_gap_power.legend()
        st.pyplot(fig_gap_power)

        gap_df = pd.DataFrame({
            "Gap_um": d["gap_vec"],
            "Kappa_um_inv": d["kappa_vec"],
            "P_cross_percent": d["p_cross_vec"],
            "P_bar_percent": d["p_bar_vec"],
            "L_residual_um": d["l_residual_vec"],
            "L_total_um": d["l_total_vec"],
            "Neff_Even": d["neff_even"],
            "Neff_Odd": d["neff_odd"],
        })
        st.dataframe(gap_df, use_container_width=True)

        csv_gap = gap_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Download Gap Sweep CSV",
            data=csv_gap,
            file_name="gap_sweep_results.csv",
            mime="text/csv",
            use_container_width=True
        )

        plt.close(fig_gap_kappa)
        plt.close(fig_gap_power)
        st.stop()

    # --- EXISTING WAVELENGTH MODE (UNCHANGED) ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Central Coupling (κ)", f"{d['kappa_vec'][d['idx_center']]:.4f} μm⁻¹")
    m2.metric("Residual Length (L_res)", f"{d['l_residual_vec'][d['idx_center']]:.2f} μm")
    m3.metric("Cross Power Transferred", f"{d['p_cross_vec'][d['idx_center']]:.1f} %")
    m4.metric(f"Q_L (at α = {d['alpha_db_vals'][1]} dB/cm)", f"{d['QL_vals'][1]/1e3:.1f} k")

    st.markdown("---")
    st.subheader("📥 Export Data & Full Report")
    
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
    ax_disp.plot(d['lambda_vec'], d['neff_even'], 'bo-', lw=2, label='n_eff Even')
    ax_disp.plot(d['lambda_vec'], d['neff_odd'], 'r^-', lw=2, label='n_eff Odd')
    ax_disp.grid(True)
    ax_disp.legend()
    ax_disp.set_xlabel('Wavelength [μm]')
    ax_disp.set_ylabel('Effective Index (n_eff)')
    ax_disp.set_title("Supermode Dispersion Curves")

    # 6. Kappa & L_res
    fig_kappa, ax_kappa_left = plt.subplots(figsize=(6, 4))
    ax_kappa_right = ax_kappa_left.twinx()
    ax_kappa_left.plot(d['lambda_vec'], d['kappa_vec'], 'kd-', lw=2, label='Kappa')
    ax_kappa_right.plot(d['lambda_vec'], d['l_residual_vec'], 'ms-', lw=2, label='L_residual')
    ax_kappa_left.grid(True)
    ax_kappa_left.set_xlabel('Wavelength [μm]')
    ax_kappa_left.set_ylabel('κ [μm⁻¹]', color='k')
    ax_kappa_right.set_ylabel('L_residual [μm]', color='m')
    ax_kappa_left.set_title("Coupling Coefficient κ & Residual Length")

    # 7. Power Transfer
    fig_power, ax_power = plt.subplots(figsize=(7, 4))
    ax_power.plot(d['lambda_vec'], d['p_cross_vec'], 'ro-', lw=2, label='Cross Port Power')
    ax_power.plot(d['lambda_vec'], d['p_bar_vec'], 'bo-', lw=2, label='Bar Port Power')
    ax_power.grid(True)
    ax_power.set_ylim(0, 105)
    ax_power.set_xlabel('Wavelength [μm]')
    ax_power.set_ylabel('Power Transfer [%]')
    ax_power.legend()
    ax_power.set_title("Power Transfer Ratio vs. Wavelength")

    # 8. Loss & Q_L
    fig_loss, ax_loss = plt.subplots(figsize=(7, 4))
    ax_loss.plot(d['lambda_vec'], d['p_cross_vec'], 'ro-', lw=2.5, label='Coupled Power P_cross')
    colors_list = ['g--', 'm--', 'k--']
    for k in range(3):
        loss_v = d['round_trip_loss_pct'][k]
        ql_v = d['QL_vals'][k] / 1e3
        alpha_db = d['alpha_db_vals'][k]
        label_text = f"Loss = {loss_v:.3f}% (α={alpha_db}dB/cm, QL≈{ql_v:.1f}k)"
        ax_loss.axhline(loss_v, color=colors_list[k][0], linestyle='--', lw=1.8, label=label_text)
    ax_loss.grid(True)
    ax_loss.set_xlabel('Wavelength [μm]')
    ax_loss.set_ylabel('Power [%]')
    ax_loss.legend(fontsize=8)
    ax_loss.set_title(f"Ring Coupling vs. Loss & Critical Q_L (L_ring = {d['L_ring_um']:.1f} μm)")

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
