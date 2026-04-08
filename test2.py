"""
Lorenz System: Trajectory Overlay on Phi-field Heatmap (Quick Fix)
This script loads pre-computed grid data and generates the trajectory overlay figure.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================================
# Part 1: Load Pre-computed Grid Data
# ============================================================================

# Load the highest resolution grid data (40x40)
try:
    data = np.load('lorenz_phi_grid_40x40.npz')
    phi_grid = data['phi_grid']
    x_edges = data['x_edges']
    z_edges = data['z_edges']
    count_grid = data['count_grid']
    print("Loaded: lorenz_phi_grid_40x40.npz")
    print(f"  phi_grid shape: {phi_grid.shape}")
    print(f"  x range: [{x_edges[0]:.1f}, {x_edges[-1]:.1f}]")
    print(f"  z range: [{z_edges[0]:.1f}, {z_edges[-1]:.1f}]")
except FileNotFoundError:
    print("Error: lorenz_phi_grid_40x40.npz not found!")
    print("Please run the main convergence study first to generate this file.")
    exit(1)

# ============================================================================
# Part 2: Generate Lorenz Trajectory
# ============================================================================

def lorenz_system(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Lorenz 63 system."""
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

print("\nGenerating Lorenz trajectory...")

# Generate a long trajectory
t_span = (0, 100)
t_eval = np.linspace(0, 100, 50000)  # 50000 points for smooth curve
initial_state = [1.0, 1.0, 1.0]

sol = solve_ivp(lorenz_system, t_span, initial_state, 
                method='RK45', t_eval=t_eval, 
                rtol=1e-8, atol=1e-10)

trajectory = sol.y.T  # shape (n_points, 3)
print(f"  Generated {len(trajectory)} points")
print(f"  x range: [{np.min(trajectory[:,0]):.1f}, {np.max(trajectory[:,0]):.1f}]")
print(f"  z range: [{np.min(trajectory[:,2]):.1f}, {np.max(trajectory[:,2]):.1f}]")

# ============================================================================
# Part 3: Plot Trajectory Overlay on Phi-field Heatmap
# ============================================================================

print("\nGenerating visualizations...")

# Figure 1: Trajectory overlay on heatmap
fig1, ax1 = plt.subplots(figsize=(14, 10))

# Compute log10(phi) for heatmap
with np.errstate(divide='ignore'):
    log_phi = np.log10(phi_grid)
masked_log_phi = np.ma.masked_where(np.isnan(log_phi), log_phi)

# Set color limits based on data percentiles
finite_vals = masked_log_phi.compressed()
if len(finite_vals) > 0:
    vmin = np.percentile(finite_vals, 5)
    vmax = np.percentile(finite_vals, 95)
else:
    vmin, vmax = -10, -5

# Plot heatmap
im1 = ax1.imshow(masked_log_phi.T, origin='lower',
                 extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]],
                 aspect='auto', cmap='hot', alpha=0.8, 
                 vmin=vmin, vmax=vmax)

# Overlay trajectory (subsampled for visual clarity)
subsample = np.arange(0, len(trajectory), 100)  # Take every 100th point
ax1.plot(trajectory[subsample, 0], trajectory[subsample, 2], 
         color='cyan', linestyle='-', linewidth=0.8, alpha=0.9, label='Lorenz trajectory')

# Mark the starting point
ax1.plot(trajectory[0, 0], trajectory[0, 2], 
         'go', markersize=8, label='Start point', alpha=0.9)

ax1.set_xlabel('x', fontsize=14)
ax1.set_ylabel('z', fontsize=14)
ax1.set_title(r'Lorenz Trajectory on $\phi$-field Heatmap (40×40 resolution)', 
              fontsize=16)
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label(r'$\log_{10}\phi$', fontsize=12)
ax1.legend(loc='upper right', fontsize=12)
ax1.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()
plt.savefig('lorenz_trajectory_overlay.png', dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: lorenz_trajectory_overlay.png")

# ============================================================================
# Part 4: Plot Trajectory Alone (for reference)
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(12, 8))

# Plot full trajectory (subsampled)
subsample2 = np.arange(0, len(trajectory), 50)
ax2.plot(trajectory[subsample2, 0], trajectory[subsample2, 2], 
         color='blue', linestyle='-', linewidth=0.6, alpha=0.8)

ax2.set_xlabel('x', fontsize=14)
ax2.set_ylabel('z', fontsize=14)
ax2.set_title('Lorenz Attractor Trajectory (x-z projection)', fontsize=16)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('lorenz_trajectory_alone.png', dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: lorenz_trajectory_alone.png")

# ============================================================================
# Part 5: Plot Heatmap Only (for reference)
# ============================================================================

fig3, ax3 = plt.subplots(figsize=(12, 10))

im3 = ax3.imshow(masked_log_phi.T, origin='lower',
                 extent=[x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]],
                 aspect='auto', cmap='hot', vmin=vmin, vmax=vmax)

ax3.set_xlabel('x', fontsize=14)
ax3.set_ylabel('z', fontsize=14)
ax3.set_title(r'$\phi$-field Heatmap (40×40 resolution)', fontsize=16)
cbar3 = plt.colorbar(im3, ax=ax3)
cbar3.set_label(r'$\log_{10}\phi$', fontsize=12)
ax3.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()
plt.savefig('lorenz_heatmap_only.png', dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: lorenz_heatmap_only.png")

# ============================================================================
# Part 6: Print Summary Statistics
# ============================================================================

print("\n" + "=" * 70)
print("Summary Statistics (40x40 Grid)")
print("=" * 70)

finite_phi = phi_grid[~np.isnan(phi_grid)]
finite_log_phi = np.log10(finite_phi)

print(f"\n  Phi-field statistics:")
print(f"    Number of valid cells: {len(finite_phi)}")
print(f"    Phi range: [{np.min(finite_phi):.2e}, {np.max(finite_phi):.2e}]")
print(f"    log10(Phi) mean: {np.mean(finite_log_phi):.3f}")
print(f"    log10(Phi) std:  {np.std(finite_log_phi):.3f}")
print(f"    log10(Phi) median: {np.median(finite_log_phi):.3f}")
print(f"    log10(Phi) 5%: {np.percentile(finite_log_phi, 5):.3f}")
print(f"    log10(Phi) 95%: {np.percentile(finite_log_phi, 95):.3f}")

print("\n" + "=" * 70)
print("All figures saved successfully!")
print("=" * 70)