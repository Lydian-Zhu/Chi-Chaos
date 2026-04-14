"""
chi-field: Publication-Quality Figures with Professional Color Schemes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Professional Color Schemes
# ============================================================================

# Color maps
CMAP_PHI = 'plasma'      # Plasma for chaos intensity (purple-gold, high contrast)
CMAP_FTLE = 'viridis'    # Viridis for FTLE (green-blue, perceptually uniform)
CMAP_PHI_LOG = 'inferno' # Inferno for log-scale (dark red-bright yellow)

# Trajectory colors
COLOR_TRAJ = '#00CED1'   # Turquoise
COLOR_TRAJ_DUFFING = '#00CED1'

# Fixed point colors
COLOR_SADDLE = '#DC143C'  # Crimson red
COLOR_CENTER = '#32CD32'  # Lime green
COLOR_START = '#FFD700'   # Gold

# Arrow colors
COLOR_ARROW = '#FFFFFF'   # White
COLOR_ARROW_EDGE = '#333333'

# Background
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.15
plt.rcParams['grid.linestyle'] = '--'


def clean_data(arr):
    """Remove NaN and Inf from array"""
    arr = np.array(arr)
    finite_mask = np.isfinite(arr)
    if np.any(finite_mask):
        min_val = np.min(arr[finite_mask])
        max_val = np.max(arr[finite_mask])
        arr = np.clip(arr, min_val, max_val)
        arr = np.nan_to_num(arr, nan=min_val, posinf=max_val, neginf=min_val)
    else:
        arr = np.zeros_like(arr)
    return arr


def get_valid_contour_levels(data, n_levels=3):
    """Get valid increasing contour levels"""
    valid_data = data[~np.isnan(data) & ~np.isinf(data)]
    if len(valid_data) < n_levels:
        return None
    levels = np.percentile(valid_data, [25, 50, 75])
    # Ensure strictly increasing
    levels = np.unique(levels)
    if len(levels) < 2:
        return None
    return levels


def load_data():
    """Load all necessary data"""
    print("Loading data...")
    data = {}
    
    # Lorenz data
    try:
        phi_lorenz = np.load('lorenz_phi_grid.npy')
        ftle_lorenz = np.load('lorenz_ftle_grid.npy')
        traj = np.load('lorenz_trajectory.npy')
        
        n_z, n_x = phi_lorenz.shape
        x_range = np.linspace(-18, 18, n_x)
        z_range = np.linspace(5, 45, n_z)
        grid_x, grid_z = np.meshgrid(x_range, z_range)
        
        data['phi_lorenz'] = phi_lorenz
        data['ftle_lorenz'] = ftle_lorenz
        data['traj'] = traj
        data['grid_x'] = grid_x
        data['grid_z'] = grid_z
        print("  Loaded Lorenz data")
    except Exception as e:
        print(f"  Lorenz data error: {e}")
    
    # Duffing data
    try:
        phi_duffing = np.load('duffing_phi_grid.npy')
        grid_x_d = np.load('duffing_grid_x.npy')
        grid_v_d = np.load('duffing_grid_v.npy')
        poincare = np.load('duffing_poincare.npy')
        x_poinc, v_poinc = poincare[:, 0], poincare[:, 1]
        
        data['phi_duffing'] = phi_duffing
        data['grid_x_d'] = grid_x_d
        data['grid_v_d'] = grid_v_d
        data['x_poinc'] = x_poinc
        data['v_poinc'] = v_poinc
        print("  Loaded Duffing data")
    except Exception as e:
        print(f"  Duffing data error: {e}")
    
    # Ordered system data
    try:
        ordered_dist = np.load('duffing_ordered_distances.npy')
        data['ordered_dist'] = ordered_dist
        data['t_eval'] = np.arange(len(ordered_dist)) * 0.02
        print("  Loaded ordered system data")
    except Exception as e:
        print(f"  Ordered data error: {e}")
    
    return data


def fig1_lorenz_phi(data):
    """Figure 1: Lorenz Phi field heatmap"""
    print("  Generating Fig1: Lorenz_Phi_magnitude.png")
    
    phi_grid = data['phi_lorenz']
    grid_x = data['grid_x']
    grid_z = data['grid_z']
    traj = data['traj']
    
    phi_grid = clean_data(phi_grid)
    log_phi = np.log10(phi_grid + 1e-10)
    log_phi = clean_data(log_phi)
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    # Smooth contour with plasma colormap
    levels = np.linspace(log_phi.min(), log_phi.max(), 40)
    im = ax.contourf(grid_x, grid_z, log_phi, levels=levels, 
                     cmap=CMAP_PHI_LOG, alpha=0.95)
    
    # Add contour lines for structure (skip if invalid)
    contour_levels = get_valid_contour_levels(log_phi, n_levels=3)
    if contour_levels is not None:
        ax.contour(grid_x, grid_z, log_phi, levels=contour_levels,
                   colors='white', linewidths=0.8, alpha=0.4, linestyles='--')
    
    # Trajectory
    ax.plot(traj[:, 0], traj[:, 2], color=COLOR_TRAJ, linewidth=1.0, alpha=0.7)
    ax.scatter(traj[0, 0], traj[0, 2], c=COLOR_START, s=100, marker='o',
               edgecolors='black', linewidth=1.5, zorder=10, label='Start')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(r'$\log_{10}\Phi$', fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('z', fontsize=14)
    ax.set_title(r'Chaos Intensity $\Phi = \|\vec{\phi}\|$ on Lorenz Attractor', 
                 fontsize=14, pad=10)
    ax.set_xlim(grid_x.min(), grid_x.max())
    ax.set_ylim(grid_z.min(), grid_z.max())
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('Lorenz_Phi_magnitude.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def fig2_lorenz_comparison(data):
    """Figure 2: FTLE vs Phi comparison"""
    print("  Generating Fig2: Lorenz_comparison.png")
    
    phi_grid = data['phi_lorenz']
    ftle_grid = data['ftle_lorenz']
    grid_x = data['grid_x']
    grid_z = data['grid_z']
    traj = data['traj']
    
    phi_grid = clean_data(phi_grid)
    ftle_grid = clean_data(ftle_grid)
    
    log_phi = np.log10(phi_grid + 1e-10)
    log_phi = clean_data(log_phi)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    
    # FTLE panel
    ax1 = axes[0]
    levels_ftle = np.linspace(ftle_grid.min(), ftle_grid.max(), 40)
    im1 = ax1.contourf(grid_x, grid_z, ftle_grid, levels=levels_ftle, 
                       cmap=CMAP_FTLE, alpha=0.95)
    ax1.plot(traj[:, 0], traj[:, 2], 'w-', linewidth=0.6, alpha=0.4)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('z', fontsize=12)
    ax1.set_title(r'(a) FTLE Field ($\tau=3.0$)', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.85)
    cbar1.set_label(r'FTLE $\lambda$', fontsize=11)
    
    # Phi panel
    ax2 = axes[1]
    im2 = ax2.contourf(grid_x, grid_z, log_phi, levels=40, 
                       cmap=CMAP_PHI_LOG, alpha=0.95)
    ax2.plot(traj[:, 0], traj[:, 2], 'c-', linewidth=0.6, alpha=0.4)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('z', fontsize=12)
    ax2.set_title(r'(b) $\Phi = \|\vec{\phi}\|$ Field', fontsize=12)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.85)
    cbar2.set_label(r'$\log_{10}\Phi$', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('Lorenz_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def fig3_lorenz_flow(data):
    """Figure 3: Lorenz phase flow overlay"""
    print("  Generating Fig3: Lorenz_phi_with_flow.png")
    
    phi_grid = data['phi_lorenz']
    grid_x = data['grid_x']
    grid_z = data['grid_z']
    traj = data['traj']
    
    phi_grid = clean_data(phi_grid)
    log_phi = np.log10(phi_grid + 1e-10)
    log_phi = clean_data(log_phi)
    
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    
    # Background
    levels = np.linspace(log_phi.min(), log_phi.max(), 40)
    im = ax.contourf(grid_x, grid_z, log_phi, levels=levels, 
                     cmap=CMAP_PHI_LOG, alpha=0.85)
    
    # Vector field
    step = 6
    X_q = grid_x[::step, ::step]
    Z_q = grid_z[::step, ::step]
    
    sigma, beta = 10.0, 8.0/3.0
    u = sigma * (0 - X_q)
    w = X_q * 0 - beta * Z_q
    
    speed = np.sqrt(u**2 + w**2)
    speed_max = np.max(speed[speed > 0])
    if speed_max > 0:
        u = u / speed_max
        w = w / speed_max
    
    ax.quiver(X_q, Z_q, u, w, color=COLOR_ARROW, alpha=0.65, scale=20,
              width=0.008, headwidth=4, headlength=5,
              edgecolors=COLOR_ARROW_EDGE, linewidth=0.5)
    
    # Trajectory
    ax.plot(traj[:, 0], traj[:, 2], color=COLOR_TRAJ, linewidth=1.2, alpha=0.8)
    ax.scatter(traj[0, 0], traj[0, 2], c=COLOR_START, s=120, marker='o',
               edgecolors='black', linewidth=1.5, zorder=10, label='Start')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(r'$\log_{10}\Phi$', fontsize=13)
    
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('z', fontsize=14)
    ax.set_title(r'Phase Flow on Chaos Intensity Field (Lorenz)', fontsize=14)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('Lorenz_phi_with_flow.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def fig4_duffing_phi(data):
    """Figure 4: Duffing Phi field"""
    print("  Generating Fig4: Duffing_chaotic_phi.png")
    
    phi_grid = data['phi_duffing']
    grid_x = data['grid_x_d']
    grid_v = data['grid_v_d']
    x_poinc = data['x_poinc']
    v_poinc = data['v_poinc']
    
    phi_grid = clean_data(phi_grid)
    log_phi = np.log10(phi_grid + 1e-10)
    log_phi = clean_data(log_phi)
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    levels = np.linspace(log_phi.min(), log_phi.max(), 40)
    im = ax.contourf(grid_x, grid_v, log_phi, levels=levels, 
                     cmap=CMAP_PHI_LOG, alpha=0.95)
    
    # Add contour lines (skip if invalid)
    contour_levels = get_valid_contour_levels(log_phi, n_levels=3)
    if contour_levels is not None:
        ax.contour(grid_x, grid_v, log_phi, levels=contour_levels,
                   colors='white', linewidths=0.6, alpha=0.4, linestyles='--')
    
    # Poincaré section
    ax.scatter(x_poinc[::20], v_poinc[::20], c='cyan', s=2, alpha=0.25, rasterized=True)
    
    # Fixed points
    ax.scatter(0, 0, c=COLOR_SADDLE, s=120, marker='X', edgecolors='white', 
               linewidth=2.5, zorder=10, label='Saddle')
    ax.scatter(1, 0, c=COLOR_CENTER, s=100, marker='o', edgecolors='white', 
               linewidth=2, zorder=10, label='Stable centers')
    ax.scatter(-1, 0, c=COLOR_CENTER, s=100, marker='o', edgecolors='white', 
               linewidth=2, zorder=10)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(r'$\log_{10}\Phi$', fontsize=13)
    
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('v', fontsize=14)
    ax.set_title(r'Chaos Intensity $\Phi = \|\vec{\phi}\|$ on Duffing Attractor', 
                 fontsize=14)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    
    plt.tight_layout()
    plt.savefig('Duffing_chaotic_phi.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def fig5_duffing_flow(data):
    """Figure 5: Duffing phase flow overlay"""
    print("  Generating Fig5: Duffing_phi_with_flow.png")
    
    phi_grid = data['phi_duffing']
    grid_x = data['grid_x_d']
    grid_v = data['grid_v_d']
    
    phi_grid = clean_data(phi_grid)
    log_phi = np.log10(phi_grid + 1e-10)
    log_phi = clean_data(log_phi)
    
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    
    levels = np.linspace(log_phi.min(), log_phi.max(), 40)
    im = ax.contourf(grid_x, grid_v, log_phi, levels=levels, 
                     cmap=CMAP_PHI_LOG, alpha=0.85)
    
    # Vector field
    step = 6
    X_q = grid_x[::step, ::step]
    V_q = grid_v[::step, ::step]
    
    delta, gamma = 0.15, 0.3
    u = V_q
    w = X_q - X_q**3 - delta * V_q + gamma
    
    speed = np.sqrt(u**2 + w**2)
    speed_max = np.max(speed[speed > 0])
    if speed_max > 0:
        u = u / speed_max
        w = w / speed_max
    
    ax.quiver(X_q, V_q, u, w, color=COLOR_ARROW, alpha=0.65, scale=20,
              width=0.008, headwidth=4, headlength=5,
              edgecolors=COLOR_ARROW_EDGE, linewidth=0.5)
    
    # Generate chaotic trajectory
    def chaotic_duffing(state, t, delta=0.15, gamma=0.3, omega=1):
        x, v = state
        return [v, x - x**3 - delta * v + gamma * np.cos(omega * t)]
    
    t_chaos = np.arange(0, 500, 0.02)
    traj_chaos = odeint(chaotic_duffing, [0, -0.1], t_chaos)
    
    # Plot trajectory
    subsample = 5
    ax.plot(traj_chaos[::subsample, 0], traj_chaos[::subsample, 1], 
            color=COLOR_TRAJ, linewidth=0.8, alpha=0.5)
    ax.scatter(traj_chaos[0, 0], traj_chaos[0, 1], c=COLOR_START, s=100, marker='o',
               edgecolors='black', linewidth=1.5, zorder=10, label='Start')
    
    # Fixed points
    ax.scatter(0, 0, c=COLOR_SADDLE, s=120, marker='X', edgecolors='white', 
               linewidth=2.5, zorder=10, label='Saddle')
    ax.scatter(1, 0, c=COLOR_CENTER, s=100, marker='o', edgecolors='white', 
               linewidth=2, zorder=10, label='Stable centers')
    ax.scatter(-1, 0, c=COLOR_CENTER, s=100, marker='o', edgecolors='white', linewidth=2, zorder=10)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(r'$\log_{10}\Phi$', fontsize=13)
    
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('v', fontsize=14)
    ax.set_title(r'Phase Flow on Chaos Intensity Field (Duffing)', fontsize=14)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    
    plt.tight_layout()
    plt.savefig('Duffing_phi_with_flow.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def fig6_duffing_ordered(data):
    """Figure 6: Ordered Duffing verification"""
    print("  Generating Fig6: Duffing_ordered.png")
    
    ordered_dist = data['ordered_dist']
    t_eval = data['t_eval']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    
    # Phase portrait
    ax1 = axes[0]
    def ordered_duffing(state, t, delta=0.3, gamma=0, omega=1):
        x, v = state
        return [v, x - x**3 - delta * v + gamma * np.cos(omega * t)]
    
    t_plot = np.arange(0, 100, 0.01)
    traj_ordered = odeint(ordered_duffing, [0.5, 0], t_plot)
    ax1.plot(traj_ordered[:, 0], traj_ordered[:, 1], 'b-', linewidth=1.2, alpha=0.8)
    ax1.scatter(traj_ordered[0, 0], traj_ordered[0, 1], c=COLOR_START, s=60, marker='o',
               edgecolors='black', zorder=10, label='Start')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('v', fontsize=12)
    ax1.set_title('(a) Phase Portrait ($\delta=0.3$, $\gamma=0$)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Error evolution
    ax2 = axes[1]
    ax2.semilogy(t_eval, ordered_dist, 'r-', linewidth=1.5, alpha=0.8)
    ax2.axhline(y=0.5, color='g', linestyle='--', linewidth=1.5, 
                label=r'Tolerance $\Delta=0.5$')
    ax2.set_xlabel('Time $t$', fontsize=12)
    ax2.set_ylabel(r'Distance $\|R(t) - q_{\text{true}}\|$', fontsize=12)
    ax2.set_title('(b) Error Contraction', fontsize=12)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Duffing_ordered.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def main():
    print("=" * 60)
    print("Generating Publication Figures with Professional Colors")
    print("=" * 60)
    
    data = load_data()
    
    print("\nGenerating figures...")
    
    if 'phi_lorenz' in data:
        fig1_lorenz_phi(data)
        fig2_lorenz_comparison(data)
        fig3_lorenz_flow(data)
    
    if 'phi_duffing' in data:
        fig4_duffing_phi(data)
        fig5_duffing_flow(data)
    
    if 'ordered_dist' in data:
        fig6_duffing_ordered(data)
    
    print("\n" + "=" * 60)
    print("All figures generated!")
    print("=" * 60)
    print("""
    Color schemes used:
    - Chaos intensity: Inferno (dark red-bright yellow)
    - FTLE: Viridis (green-blue, perceptually uniform)
    - Trajectories: Turquoise (#00CED1)
    - Start point: Gold (#FFD700)
    - Saddle point: Crimson (#DC143C)
    - Stable centers: Lime green (#32CD32)
    - Flow arrows: White with dark edges
    """)


if __name__ == "__main__":
    main()