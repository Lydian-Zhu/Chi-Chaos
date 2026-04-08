"""
Lorenz System: Grid Convergence Study for Phi-field Heatmap (FIXED)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Part 1: Lorenz System
# ============================================================================

def lorenz_system(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])


# ============================================================================
# Part 2: Forecast Time Computation
# ============================================================================

def compute_forecast_time_fast(q0, q_pert, t_max=50.0, rtol=1e-6, atol=1e-8, threshold=1.0):
    """Compute forecast time T: first time when ||q - q_pert|| > threshold."""
    state0 = np.concatenate([q0, q_pert])
    
    def combined_system(t, state):
        return np.concatenate([lorenz_system(t, state[:3]), 
                               lorenz_system(t, state[3:])])
    
    def threshold_event(t, state):
        return np.linalg.norm(state[:3] - state[3:]) - threshold
    threshold_event.terminal = True
    threshold_event.direction = 1
    
    sol = solve_ivp(combined_system, (0, t_max), state0, 
                    method='RK45', rtol=rtol, atol=atol,
                    events=threshold_event)
    
    if sol.t_events[0].size > 0:
        return sol.t_events[0][0]
    else:
        return t_max


# ============================================================================
# Part 3: Chi and Phi Computation
# ============================================================================

def compute_chi_at_point(q0, eps0=1e-8, delta=1e-9, 
                         t_max=50.0, rtol=1e-6, atol=1e-8, 
                         threshold=1.0):
    """Compute chi vector at fixed epsilon = eps0."""
    chi = np.zeros(3)
    
    for i in range(3):
        e_i = np.zeros(3)
        e_i[i] = 1.0
        
        q_plus = q0 + (eps0 + delta) * e_i
        q_minus = q0 + (eps0 - delta) * e_i
        
        T_plus = compute_forecast_time_fast(q0, q_plus, t_max, rtol, atol, threshold)
        T_minus = compute_forecast_time_fast(q0, q_minus, t_max, rtol, atol, threshold)
        
        chi[i] = (T_plus - T_minus) / (2.0 * delta)
    
    return chi


def compute_phi_at_point(q0, eps0=1e-8, delta=1e-9, 
                         t_max=50.0, rtol=1e-6, atol=1e-8, 
                         threshold=1.0):
    """Compute phi = 1/||chi|| at a point."""
    chi = compute_chi_at_point(q0, eps0, delta, t_max, rtol, atol, threshold)
    norm_chi = np.linalg.norm(chi)
    
    if norm_chi < 1e-15 or not np.isfinite(norm_chi):
        return np.inf
    else:
        return 1.0 / norm_chi


# ============================================================================
# Part 4: Generate Trajectory
# ============================================================================

def generate_attractor_trajectory(n_steps=10000, dt=0.01, burn_in=2000):
    """Generate a long trajectory on the Lorenz attractor."""
    state = np.array([1.0, 1.0, 1.0])
    
    # Burn-in
    for _ in range(burn_in):
        k1 = lorenz_system(0, state)
        k2 = lorenz_system(0, state + 0.5 * dt * k1)
        k3 = lorenz_system(0, state + 0.5 * dt * k2)
        k4 = lorenz_system(0, state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Sampling
    trajectory = []
    for _ in range(n_steps):
        for _ in range(10):
            k1 = lorenz_system(0, state)
            k2 = lorenz_system(0, state + 0.5 * dt * k1)
            k3 = lorenz_system(0, state + 0.5 * dt * k2)
            k4 = lorenz_system(0, state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        trajectory.append(state.copy())
    
    return np.array(trajectory)


# ============================================================================
# Part 5: Grid-based Phi Field Computation
# ============================================================================

def compute_phi_grid(trajectory, resolution, 
                      eps0=1e-8, delta=1e-9,
                      t_max=50.0, threshold=1.0,
                      max_samples_per_cell=5,
                      rtol=1e-6, atol=1e-8):
    """
    Compute phi grid at a given resolution.
    
    Returns
    -------
    phi_grid : 2D array
        Average phi value per cell (NaN for empty cells)
    x_edges, z_edges : 1D arrays
        Bin edges
    count_grid : 2D array
        Number of samples per cell
    """
    x = trajectory[:, 0]
    z = trajectory[:, 2]
    
    x_range = (np.min(x), np.max(x))
    z_range = (np.min(z), np.max(z))
    
    # Add small padding
    x_pad = 0.05 * (x_range[1] - x_range[0])
    z_pad = 0.05 * (z_range[1] - z_range[0])
    x_edges = np.linspace(x_range[0] - x_pad, x_range[1] + x_pad, resolution + 1)
    z_edges = np.linspace(z_range[0] - z_pad, z_range[1] + z_pad, resolution + 1)
    
    # Initialize
    phi_grid = np.full((resolution, resolution), np.nan)
    count_grid = np.zeros((resolution, resolution))
    
    # Assign points to cells
    cell_indices = {}
    for idx, (xi, zi) in enumerate(zip(x, z)):
        ix = np.digitize(xi, x_edges) - 1
        iz = np.digitize(zi, z_edges) - 1
        if 0 <= ix < resolution and 0 <= iz < resolution:
            key = (ix, iz)
            if key not in cell_indices:
                cell_indices[key] = []
            if len(cell_indices[key]) < max_samples_per_cell:
                cell_indices[key].append(idx)
    
    print(f"    Resolution {resolution}x{resolution}: {len(cell_indices)} non-empty cells")
    
    # Compute phi for each cell
    for (ix, iz), indices in tqdm(cell_indices.items(), desc=f"    Computing res={resolution}"):
        phis = []
        for idx in indices:
            q0 = trajectory[idx]
            try:
                phi = compute_phi_at_point(q0, eps0, delta, t_max, rtol, atol, threshold)
                if np.isfinite(phi) and phi > 0 and phi < 1e10:  # Sanity check
                    phis.append(phi)
            except Exception as e:
                continue
        
        if len(phis) >= 1:
            phi_grid[ix, iz] = np.mean(phis)
            count_grid[ix, iz] = len(phis)
    
    return phi_grid, x_edges, z_edges, count_grid


# ============================================================================
# Part 6: Visualization
# ============================================================================

def plot_heatmaps(results, save_prefix='lorenz_phi'):
    """Plot heatmaps for all resolutions."""
    resolutions = sorted(results.keys())
    n_res = len(resolutions)
    
    fig, axes = plt.subplots(1, n_res, figsize=(5*n_res, 4.5))
    if n_res == 1:
        axes = [axes]
    
    for idx, res in enumerate(resolutions):
        ax = axes[idx]
        data = results[res]
        phi_grid = data['phi_grid']
        x_edges = data['x_edges']
        z_edges = data['z_edges']
        
        # Use log10 for phi
        with np.errstate(divide='ignore'):
            log_phi = np.log10(phi_grid)
        masked_log_phi = np.ma.masked_where(np.isnan(log_phi), log_phi)
        
        # Set reasonable color limits
        finite_vals = masked_log_phi.compressed()
        if len(finite_vals) > 0:
            vmin = np.percentile(finite_vals, 5)
            vmax = np.percentile(finite_vals, 95)
        else:
            vmin, vmax = -10, -5
        
        im = ax.imshow(masked_log_phi.T, origin='lower',
                       extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]],
                       aspect='auto', cmap='hot', vmin=vmin, vmax=vmax)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('z', fontsize=12)
        ax.set_title(f'{res}x{res}', fontsize=14)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r'$\log_{10}\phi$', fontsize=10)
    
    plt.suptitle(r'$\phi$-field Heatmap Convergence Study', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_prefix}_heatmaps.png")


def plot_convergence_metrics(results, save_prefix='lorenz_phi'):
    """Plot convergence metrics."""
    resolutions = sorted(results.keys())
    
    means, stds, medians, q05, q95 = [], [], [], [], []
    
    for res in resolutions:
        phi_grid = results[res]['phi_grid']
        finite_phi = phi_grid[~np.isnan(phi_grid)]
        
        if len(finite_phi) > 0:
            log_phi = np.log10(finite_phi)
            means.append(np.mean(log_phi))
            stds.append(np.std(log_phi))
            medians.append(np.median(log_phi))
            q05.append(np.percentile(log_phi, 5))
            q95.append(np.percentile(log_phi, 95))
        else:
            means.append(np.nan)
            stds.append(np.nan)
            medians.append(np.nan)
            q05.append(np.nan)
            q95.append(np.nan)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mean and std
    ax1 = axes[0]
    ax1.errorbar(resolutions, means, yerr=stds, marker='o', capsize=5,
                 color='black', ecolor='gray', linewidth=2)
    ax1.set_xlabel('Grid Resolution', fontsize=12)
    ax1.set_ylabel(r'Mean $\log_{10}\phi$', fontsize=12)
    ax1.set_title('Mean and Standard Deviation', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Quantiles
    ax2 = axes[1]
    ax2.fill_between(resolutions, q05, q95, alpha=0.3, color='gray', label='5%-95%')
    ax2.plot(resolutions, medians, 'r-', linewidth=2, label='Median')
    ax2.set_xlabel('Grid Resolution', fontsize=12)
    ax2.set_ylabel(r'$\log_{10}\phi$', fontsize=12)
    ax2.set_title('Median and 5%-95% Range', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_prefix}_metrics.png")
    
    # Print table
    print("\n" + "=" * 70)
    print("Convergence Metrics")
    print("=" * 70)
    print(f"{'Res':>6} {'Mean':>12} {'Std':>12} {'Median':>12} {'5%':>12} {'95%':>12}")
    print("-" * 70)
    for i, res in enumerate(resolutions):
        if not np.isnan(means[i]):
            print(f"{res:>6} {means[i]:>12.3f} {stds[i]:>12.3f} {medians[i]:>12.3f} {q05[i]:>12.3f} {q95[i]:>12.3f}")
        else:
            print(f"{res:>6} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
    print("=" * 70)


def plot_trajectory_overlay(results, trajectory, save_prefix='lorenz_phi'):
    """Overlay trajectory on highest resolution heatmap."""
    best_res = max(results.keys())
    data = results[best_res]
    phi_grid = data['phi_grid']
    x_edges = data['x_edges']
    z_edges = data['z_edges']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    with np.errstate(divide='ignore'):
        log_phi = np.log10(phi_grid)
    masked_log_phi = np.ma.masked_where(np.isnan(log_phi), log_phi)
    
    finite_vals = masked_log_phi.compressed()
    if len(finite_vals) > 0:
        vmin = np.percentile(finite_vals, 5)
        vmax = np.percentile(finite_vals, 95)
    else:
        vmin, vmax = -10, -5
    
    im = ax.imshow(masked_log_phi.T, origin='lower',
                   extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]],
                   aspect='auto', cmap='hot', alpha=0.8, vmin=vmin, vmax=vmax)
    
    # Plot trajectory (subsampled)
    subsample = slice(0, len(trajectory), 500)
    ax.plot(trajectory[subsample, 0], trajectory[subsample, 2],
            'b-', alpha=0.4, linewidth=0.8, label='Trajectory')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('z', fontsize=12)
    ax.set_title(f'Lorenz Attractor on $\phi$-field (Resolution: {best_res}x{best_res})',
                 fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\log_{10}\phi$', fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_trajectory.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {save_prefix}_trajectory.png")


# ============================================================================
# Part 7: Main
# ============================================================================

def main():
    print("=" * 70)
    print("Lorenz System: Grid Convergence Study for Phi-field Heatmap")
    print("=" * 70)
    
    # Parameters - ADJUST THESE FOR FASTER/SLOWER RUN
    n_trajectory_points = 5000    # Number of points on attractor
    resolutions = [20, 30, 40]    # Grid resolutions for convergence study
    max_samples_per_cell = 3      # Max points per grid cell (keep small for speed)
    eps0 = 1e-8
    delta = 1e-9
    t_max = 30.0                  # Reduced for speed
    threshold = 1.0
    rtol = 1e-6                   # Looser tolerance for speed
    atol = 1e-8
    
    print(f"\nParameters:")
    print(f"  Trajectory points: {n_trajectory_points}")
    print(f"  Grid resolutions: {resolutions}")
    print(f"  Max samples per cell: {max_samples_per_cell}")
    print(f"  eps0 = {eps0:.1e}, delta = {delta:.1e}")
    print(f"  threshold = {threshold}, t_max = {t_max}")
    
    # Step 1: Generate trajectory
    print("\n[1/4] Generating Lorenz attractor trajectory...")
    trajectory = generate_attractor_trajectory(n_steps=n_trajectory_points)
    print(f"  Generated {len(trajectory)} points")
    print(f"  x range: [{np.min(trajectory[:,0]):.1f}, {np.max(trajectory[:,0]):.1f}]")
    print(f"  z range: [{np.min(trajectory[:,2]):.1f}, {np.max(trajectory[:,2]):.1f}]")
    
    # Step 2: Compute phi grids
    print("\n[2/4] Computing phi grids...")
    results = {}
    
    for res in resolutions:
        print(f"\n  Processing resolution: {res}x{res}")
        phi_grid, x_edges, z_edges, count_grid = compute_phi_grid(
            trajectory, res, eps0, delta, t_max, threshold, 
            max_samples_per_cell, rtol, atol
        )
        
        results[res] = {
            'phi_grid': phi_grid,
            'x_edges': x_edges,
            'z_edges': z_edges,
            'count_grid': count_grid
        }
        
        # Quick stats
        finite_phi = phi_grid[~np.isnan(phi_grid)]
        if len(finite_phi) > 0:
            print(f"    Valid cells: {len(finite_phi)}")
            print(f"    Phi range: [{np.min(finite_phi):.2e}, {np.max(finite_phi):.2e}]")
    
    # Step 3: Visualize
    print("\n[3/4] Generating visualizations...")
    plot_heatmaps(results)
    plot_convergence_metrics(results)
    plot_trajectory_overlay(results, trajectory)
    
    # Step 4: Save data
    print("\n[4/4] Saving data...")
    for res, data in results.items():
        np.savez(f'lorenz_phi_grid_{res}x{res}.npz',
                 phi_grid=data['phi_grid'],
                 x_edges=data['x_edges'],
                 z_edges=data['z_edges'],
                 count_grid=data['count_grid'])
        print(f"  Saved: lorenz_phi_grid_{res}x{res}.npz")
    
    print("\n" + "=" * 70)
    print("Convergence study complete!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()