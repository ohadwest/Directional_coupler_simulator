import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def sellmeier_sin(lam_um):
    """Sellmeier formula for Silicon Nitride (SiN)"""
    b1, c1, d = 2.938, 0.13372, 0.02573
    n_sq = 1.0 + (b1 * lam_um**2) / (lam_um**2 - c1**2) - d * lam_um**2
    return np.sqrt(n_sq)

def sellmeier_sio2(lam_um):
    """Sellmeier formula for Silica (SiO2)"""
    b1, c1 = 0.6961663, 0.0684043
    b2, c2 = 0.4079426, 0.1162414
    b3, c3 = 0.8974794, 9.896161
    n_sq = 1.0 + (b1 * lam_um**2) / (lam_um**2 - c1**2) + \
                 (b2 * lam_um**2) / (lam_um**2 - c2**2) + \
                 (b3 * lam_um**2) / (lam_um**2 - c3**2)
    return np.sqrt(n_sq)

def waveguidemeshfull(n_layers, h_layers, h_core, total_half_width, side, dx, dy):
    side1 = side2 = side
    ih = np.round(np.array(h_layers) / dy).astype(int)
    irh = int(np.round(h_core / dy))
    irw = int(np.round(2 * total_half_width / dx))
    iside1 = int(np.round(side1 / dx))
    iside2 = int(np.round(side2 / dx))
    
    nx = irw + iside1 + iside2 + 1
    ny = np.sum(ih) + 1
    
    x = dx * np.arange(-(irw // 2 + iside1), (irw // 2 + iside2) + 1)
    xc = 0.5 * (x[:-1] + x[1:])
    y = np.arange(ny) * dy
    yc = np.arange(1, ny) * dy - dy / 2.0
    
    eps = np.zeros((len(xc), len(yc)))
    iy = 0
    for jj, n_val in enumerate(n_layers):
        for _ in range(ih[jj]):
            eps[:, iy] = n_val**2
            iy += 1
            
    iy_start = np.sum(ih) - ih[-1] - 1
    for _ in range(irh):
        eps[:iside1, iy_start] = n_layers[-1]**2
        eps[irw + iside1:, iy_start] = n_layers[-1]**2
        iy_start -= 1
        
    return xc, yc, eps

def svmodes_2d(lam_um, guess, nmodes, dx, dy, eps_mesh, polarization='ex'):
    nx, ny = eps_mesh.shape
    k0 = 2.0 * np.pi / lam_um
    eps_padded = np.pad(eps_mesh, ((1, 1), (1, 1)), mode='edge')
    
    ep = eps_padded[1:nx+1, 1:ny+1]
    en = eps_padded[1:nx+1, 2:ny+2]
    es = eps_padded[1:nx+1, 0:ny]
    ee = eps_padded[2:nx+2, 1:ny+1]
    ew = eps_padded[0:nx,   1:ny+1]
    
    n_mat = np.full((nx, ny), dy)
    s_mat = np.full((nx, ny), dy)
    e_mat = np.full((nx, ny), dx)
    w_mat = np.full((nx, ny), dx)
    p_mat = np.full((nx, ny), dx)
    q_mat = np.full((nx, ny), dy)
    
    if polarization.lower() == 'ex':
        an = 2.0 / (n_mat * (n_mat + s_mat))
        as_ = 2.0 / (s_mat * (n_mat + s_mat))
        num_e = 8.0 * (p_mat * (ep - ew) + 2.0 * w_mat * ew) * ee
        den_e = (p_mat * (ep - ee) + 2.0 * e_mat * ee) * (p_mat**2 * (ep - ew) + 4.0 * w_mat**2 * ew) + \
                (p_mat * (ep - ew) + 2.0 * w_mat * ew) * (p_mat**2 * (ep - ee) + 4.0 * e_mat**2 * ee)
        ae = num_e / den_e
        num_w = 8.0 * (p_mat * (ep - ee) + 2.0 * e_mat * ee) * ew
        aw = num_w / den_e
        ap = ep * (k0**2) - an - as_ - ae * (ep / ee) - aw * (ep / ew)
    else:
        num_n = 8.0 * (q_mat * (ep - es) + 2.0 * s_mat * es) * en
        den_n = (q_mat * (ep - en) + 2.0 * n_mat * en) * (q_mat**2 * (ep - es) + 4.0 * s_mat**2 * es) + \
                (q_mat * (ep - es) + 2.0 * s_mat * es) * (q_mat**2 * (ep - en) + 4.0 * n_mat**2 * en)
        an = num_n / den_n
        as_ = 8.0 * (q_mat * (ep - en) + 2.0 * n_mat * en) * es / den_n
        ae = 2.0 / (e_mat * (e_mat + w_mat))
        aw = 2.0 / (w_mat * (e_mat + w_mat))
        ap = ep * (k0**2) - an * (ep / en) - as_ * (ep / es) - ae - aw

    N = nx * ny
    main_diag = ap.flatten('F')
    
    # Correct diagonal array alignment for 1D flattened Fortran-indexed 2D grid:
    ae_diag = ae.flatten('F')[:-1]
    aw_diag = aw.flatten('F')[1:]
    an_diag = an.flatten('F')[:-nx]
    as_diag = as_.flatten('F')[nx:]
    
    A = sp.diags([main_diag, ae_diag, aw_diag, an_diag, as_diag], [0, 1, -1, nx, -nx], shape=(N, N), format='csc')
    shift = (2.0 * np.pi * guess / lam_um)**2
    vals, vecs = spla.eigs(A, k=nmodes, sigma=shift, which='LM')
    
    neff_vals = (lam_um / (2.0 * np.pi)) * np.sqrt(np.real(vals))
    phi_modes = np.zeros((nx, ny, nmodes))
    
    for idx in range(nmodes):
        mode_2d = np.real(vecs[:, idx]).reshape((nx, ny), order='F')
        max_abs = np.max(np.abs(mode_2d))
        if max_abs > 0:
            mode_2d /= max_abs
        phi_modes[:, :, idx] = mode_2d
        
    return phi_modes, neff_vals

def run_simulation(w_single, h_core, gap, coupler_L, ring_R, lambda_start, lambda_end, n_lambda, polarization, res_mode, top_oxide):
    """Main simulation runner function"""
    dx = dy = 0.005 if "hr" in res_mode else (0.01 if "mr" in res_mode else 0.02)
    top_clad_mode = 'air' if top_oxide <= 0 else 'thin_silica'
    side = 2.0
    
    lambda_vec = np.linspace(lambda_start, lambda_end, n_lambda)
    neff_even_vec = np.zeros(n_lambda)
    neff_odd_vec = np.zeros(n_lambda)
    kappa_vec = np.zeros(n_lambda)
    l_residual_vec = np.zeros(n_lambda)
    l_total_vec = np.zeros(n_lambda)
    p_cross_vec = np.zeros(n_lambda)
    p_bar_vec = np.zeros(n_lambda)
    
    idx_center = n_lambda // 2
    
    for i in range(n_lambda):
        current_lambda = lambda_vec[i]
        k0 = 2.0 * np.pi / current_lambda
        
        n_core = sellmeier_sin(current_lambda)
        n_clad = sellmeier_sio2(current_lambda)
        
        total_half_width = w_single + gap / 2.0
        h_layers = [4.0, h_core, 2.0]
        n_layers = [n_clad, n_core, n_clad]
        
        xc, yc, eps_mesh = waveguidemeshfull(n_layers, h_layers, h_core, total_half_width, side, dx, dy)
        nx, ny = eps_mesh.shape
        
        core_mask_large = np.abs(eps_mesh - n_core**2) < 1e-5
        x_idx, y_idx = np.where(core_mask_large)
        core_top_y = yc[np.max(y_idx)] + dy / 2.0
        core_bottom_y = yc[np.min(y_idx)] - dy / 2.0
        
        gap_left, gap_right = -gap / 2.0, gap / 2.0
        gap_mask = (xc >= gap_left) & (xc <= gap_right)
        
        for col in range(ny):
            if core_bottom_y <= yc[col] <= core_top_y:
                eps_mesh[gap_mask, col] = n_clad**2
                
        if top_clad_mode == 'air':
            for col in range(ny):
                if yc[col] > core_top_y:
                    eps_mesh[:, col] = 1.0**2
        else:
            interface_y = core_top_y + top_oxide
            for col in range(ny):
                if yc[col] > interface_y:
                    eps_mesh[:, col] = 1.0**2
                    
        guess = (n_core + n_clad) / 2.0
        phi_modes, neff_vals = svmodes_2d(current_lambda, guess, 2, dx, dy, eps_mesh, polarization)
        
        sorted_indices = np.argsort(neff_vals)[::-1]
        neff_even_vec[i] = neff_vals[sorted_indices[0]]
        neff_odd_vec[i] = neff_vals[sorted_indices[1]]
        
        kappa_vec[i] = (np.pi / current_lambda) * (neff_even_vec[i] - neff_odd_vec[i])
        
        if ring_R > 0:
            n_eff_avg = (neff_even_vec[i] + neff_odd_vec[i]) / 2.0
            gamma_val = k0 * np.sqrt(max(n_eff_avg**2 - n_clad**2, 1e-4))
            l_residual_vec[i] = np.sqrt(np.pi * ring_R / gamma_val)
        else:
            l_residual_vec[i] = 0.0
            
        l_total_vec[i] = coupler_L + l_residual_vec[i]
        p_cross_vec[i] = (np.sin(kappa_vec[i] * l_total_vec[i]))**2 * 100.0
        p_bar_vec[i] = (np.cos(kappa_vec[i] * l_total_vec[i]))**2 * 100.0
        
        if i == idx_center:
            eps_mesh_center = eps_mesh.copy()
            phi_even = phi_modes[:, :, sorted_indices[0]]
            phi_odd = phi_modes[:, :, sorted_indices[1]]
            
            if np.sum(phi_even) < 0: phi_even = -phi_even
            phi_even /= np.max(np.abs(phi_even))
            
            mid_x_idx = nx // 2
            if np.sum(phi_odd[mid_x_idx:, :]) < 0: phi_odd = -phi_odd
            phi_odd /= np.max(np.abs(phi_odd))
            
            xc_center, yc_center = xc, yc
            lambda_center_val = current_lambda
            box1_l, box1_r = -gap/2.0 - w_single, -gap/2.0
            box2_l, box2_r = gap/2.0, gap/2.0 + w_single
            b_y, t_y = core_bottom_y, core_top_y
            mid_y_idx = np.argmin(np.abs(yc - (b_y + t_y) / 2.0))

    L_ring_um = (2 * np.pi * ring_R + 2 * coupler_L) if ring_R > 0 else (2 * coupler_L)
    L_ring_cm = L_ring_um * 1e-4
    alpha_db_vals = np.array([0.5, 1.5, 5.0])
    alpha_cm = alpha_db_vals * (np.log(10) / 10.0)
    round_trip_loss_pct = (1.0 - np.exp(-alpha_cm * L_ring_cm)) * 100.0
    
    neff_avg_vec = (neff_even_vec + neff_odd_vec) / 2.0
    lambda_cm_center = lambda_center_val * 1e-4
    dneff_dlambda = (neff_avg_vec[-1] - neff_avg_vec[0]) / ((lambda_vec[-1] - lambda_vec[0]) * 1e-4)
    n_group = neff_avg_vec[idx_center] - lambda_cm_center * dneff_dlambda
    
    Q0_vals = (2.0 * np.pi * n_group) / (lambda_cm_center * alpha_cm)
    QL_vals = Q0_vals / 2.0
    
    return {
        'xc': xc_center, 'yc': yc_center, 'eps_center': eps_mesh_center,
        'phi_even': phi_even, 'phi_odd': phi_odd, 'mid_y_idx': mid_y_idx,
        'lambda_vec': lambda_vec, 'neff_even': neff_even_vec, 'neff_odd': neff_odd_vec,
        'kappa_vec': kappa_vec, 'l_residual_vec': l_residual_vec, 'l_total_vec': l_total_vec,
        'p_cross_vec': p_cross_vec, 'p_bar_vec': p_bar_vec, 'round_trip_loss_pct': round_trip_loss_pct,
        'QL_vals': QL_vals, 'alpha_db_vals': alpha_db_vals, 'L_ring_um': L_ring_um,
        'lambda_center_val': lambda_center_val, 'idx_center': idx_center,
        'box1_l': box1_l, 'box1_r': box1_r, 'box2_l': box2_l, 'box2_r': box2_r,
        'b_y': b_y, 't_y': t_y, 'polarization': polarization
    }
