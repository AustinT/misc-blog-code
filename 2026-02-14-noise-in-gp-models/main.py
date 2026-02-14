"""Generates plots for the post."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# --- Configuration & Hyperparameters ---
np.random.seed(42)
n_samples = 50
grid_res = 100
true_std = 0.01 
true_var = true_std**2  # 10^-4
PLOT_DPI = 200

# Colors
color_ml = 'red'
color_fixed = 'lime'  
color_delta = 'blue'

# 1. Setup Data
X = np.sort(np.random.uniform(0, 1, n_samples)).reshape(-1, 1)
X_plot = np.linspace(0, 1, 500).reshape(-1, 1)

def true_func(x):
    # Sin(2pi x) + 0.1 * sawtooth
    return np.sin(2 * np.pi * x) + 0.2 * sawtooth(2 * np.pi * 5 * x)

y_clean = true_func(X).flatten()
y = y_clean + np.random.normal(0, true_std, size=n_samples)

# 2. Grid for Contour Plots
# EXTENDED LS range to 10^-3 to show the "edge" solution clearly
ls_range = np.logspace(-3, 0, grid_res) 
noise_range = np.logspace(-5, 0, grid_res)
LS, NS = np.meshgrid(ls_range, noise_range)

def get_lml(l, n):
    gp = GaussianProcessRegressor(kernel=1.0 * RBF(l) + WhiteKernel(n), optimizer=None)
    gp.fit(X, y)
    return gp.log_marginal_likelihood()


def gp_predictive_uncertainty(gp, X, noise_var):
    """Return GP mean, stddev of latent function f (epistemic), and stddev of predictive output (total)."""
    y_mean, y_std_total = gp.predict(X, return_std=True)
    f_var = np.maximum(y_std_total**2 - noise_var, 0)
    f_std = np.sqrt(f_var)
    return y_mean, f_std, y_std_total


def plot_gp_fit(ax, gp, X_plot, noise_var, color, label_mean="GP mean"):
    """Plot observed data, true function, GP mean, and epistemic/aleatoric ±1 std bands."""
    x_flat = X_plot.flatten()
    y_mean, f_std, y_std_total = gp_predictive_uncertainty(gp, X_plot, noise_var)

    ax.scatter(X, y, c='k', marker='x', alpha=0.6, label='Observed data')
    ax.plot(X_plot, true_func(X_plot), 'k', label='Function value')

    # y ± 1 std (epistemic + aleatoric): solid fill
    ax.fill_between(x_flat, y_mean - y_std_total, y_mean + y_std_total, color=color, alpha=0.35, label='y ± 1 std')
    # f ± 1 std (epistemic): hatched fill
    ax.fill_between(x_flat, y_mean - f_std, y_mean + f_std, color=color, alpha=0.2, hatch='///', label='f ± 1 std')

    ax.plot(X_plot, y_mean, color=color, label=label_mean)
    ax.legend(fontsize='x-small')


# --- Figure 1: Standard GP Misspecification ---
print("Computing LML grid for Figure 1...")
LML_fig1 = np.array([[get_lml(l, n) for l in ls_range] for n in noise_range])

# Find Global ML Solutions
idx_ml = np.unravel_index(np.argmax(LML_fig1), LML_fig1.shape)
ml_ls, ml_noise = ls_range[idx_ml[1]], noise_range[idx_ml[0]]

# ML Lengthscale if noise is strictly fixed at true_var
lml_fixed_noise = [get_lml(l, true_var) for l in ls_range]
fixed_noise_ls = ls_range[np.argmax(lml_fixed_noise)]

fig1, axes = plt.subplots(1, 3, figsize=(18, 5))

# Subplot 1: LML Surface
cp = axes[0].contourf(LS, NS, LML_fig1, levels=25, cmap='viridis')
fig1.colorbar(cp, ax=axes[0])
axes[0].set_xscale('log')
axes[0].set_yscale('log')

# Highlight Global ML
axes[0].scatter(ml_ls, ml_noise, color=color_ml, s=100, edgecolors='white', zorder=5, label='Global ML')
axes[0].axvline(ml_ls, color=color_ml, linestyle='--', alpha=0.5)
axes[0].axhline(ml_noise, color=color_ml, linestyle='--', alpha=0.5)

# Highlight Fixed Noise ML (The "Collapsed" solution)
axes[0].scatter(fixed_noise_ls, true_var, color=color_fixed, s=100, edgecolors='black', zorder=5, label='Fixed Noise ML')
axes[0].axvline(fixed_noise_ls, color=color_fixed, linestyle='--', alpha=0.5)
axes[0].axhline(true_var, color=color_fixed, linestyle='--', alpha=0.5)

axes[0].set_title("LML Surface: Global vs. Fixed Noise")
axes[0].set_xlabel("Lengthscale (Log Scale)")
axes[0].set_ylabel("Noise Variance (Log Scale)")
axes[0].legend(fontsize='x-small', loc='upper left')

# Subplot 2: Global ML Fit
gp_ml = GaussianProcessRegressor(kernel=1.0 * RBF(ml_ls) + WhiteKernel(ml_noise), optimizer=None).fit(X, y)
plot_gp_fit(axes[1], gp_ml, X_plot, ml_noise, color_ml, label_mean="GP mean")
y_mean_ml, _, _ = gp_predictive_uncertainty(gp_ml, X_plot, ml_noise)
x_flat = X_plot.flatten()
x_true_max = x_flat[np.argmax(true_func(X_plot))]
x_gp_max = x_flat[np.argmax(y_mean_ml)]
axes[1].axvline(x_true_max, color='k', linestyle='--', label='True max')
axes[1].axvline(x_gp_max, color=color_ml, linestyle='--', label='GP mean max')
axes[1].legend(fontsize='x-small')
axes[1].set_title(f"Global ML: High Noise\n($\ell$={ml_ls:.3f}, $\sigma^2$={ml_noise:.4f})")

# Subplot 3: Fixed True Noise Fit
gp_fix = GaussianProcessRegressor(kernel=1.0 * RBF(fixed_noise_ls) + WhiteKernel(true_var), optimizer=None).fit(X, y)
plot_gp_fit(axes[2], gp_fix, X_plot, true_var, color_fixed, label_mean="GP mean")
axes[2].set_title(f"Fixed Noise: Short Lengthscale\n($\ell$={fixed_noise_ls:.3f}, $\sigma^2$={true_var:.4f})")

plt.tight_layout()
plt.savefig("gp_misspecification_comparison.png", dpi=PLOT_DPI)

# --- Figure 2: Delta Solution ---
delta_range = np.logspace(-5, 0, grid_res)
LD, DD = np.meshgrid(ls_range, delta_range)

print("Computing LML grid for Figure 2...")
LML_fig2 = np.array([[get_lml(l, true_var + d) for l in ls_range] for d in delta_range])

idx_delta_ml = np.unravel_index(np.argmax(LML_fig2), LML_fig2.shape)
ml_ls_delta, ml_delta = ls_range[idx_delta_ml[1]], delta_range[idx_delta_ml[0]]

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

# Subplot 1: LML Surface with Delta
cp2 = axes2[0].contourf(LD, DD, LML_fig2, levels=25, cmap='magma')
fig2.colorbar(cp2, ax=axes2[0])
axes2[0].set_xscale('log')
axes2[0].set_yscale('log')
axes2[0].scatter(ml_ls_delta, ml_delta, color=color_delta, s=100, edgecolors='white', zorder=5)
axes2[0].axvline(ml_ls_delta, color=color_delta, linestyle='--', alpha=0.5)
axes2[0].axhline(ml_delta, color=color_delta, linestyle='--', alpha=0.5)
axes2[0].set_title("LML: Lengthscale vs. Delta\n(Fixed True Noise)")
axes2[0].set_xlabel("Lengthscale")
axes2[0].set_ylabel(r"$\sigma^2_m$")

# Subplot 2: Predictive Distribution
gp_delta = GaussianProcessRegressor(kernel=1.0 * RBF(ml_ls_delta) + WhiteKernel(true_var + ml_delta), optimizer=None).fit(X, y)
plot_gp_fit(axes2[1], gp_delta, X_plot, true_var, color_delta, label_mean="GP mean")
y_mean_d, _, _ = gp_predictive_uncertainty(gp_delta, X_plot, true_var + ml_delta)
x_flat = X_plot.flatten()
x_true_max = x_flat[np.argmax(true_func(X_plot))]
x_gp_max = x_flat[np.argmax(y_mean_d)]
axes2[1].axvline(x_true_max, color='k', linestyle='--', label='True max')
axes2[1].axvline(x_gp_max, color=color_delta, linestyle='--', label='GP mean max')
axes2[1].legend(fontsize='x-small')
axes2[1].set_title("Predictive Distribution with misspecification delta term")

plt.tight_layout()
plt.savefig("gp_delta_solution.png", dpi=PLOT_DPI)
