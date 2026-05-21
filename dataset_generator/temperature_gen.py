import numpy as np
from numpy.fft import rfftn, irfftn, rfftfreq
from scipy.ndimage import shift as nd_shift
from tqdm import tqdm

import numpy as np
from numpy.fft import rfftn, irfftn, rfftfreq
from scipy.ndimage import shift as nd_shift
from datetime import date, timedelta

# ------------------------------
# Configuration: 2y daily data
# ------------------------------
years = 2
start_date = date(2015, 1, 1)   # choose any start; leap years handled
# Build exact daily index (includes leap days)
dates = []
d = start_date
for _ in range(40000):  # generous upper bound
    dates.append(d)
    d = d + timedelta(days=1)
    if (d - start_date).days >= 365*years + sum(1 for y in range(start_date.year, start_date.year+years) 
                                                if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))):
        break
T_steps = len(dates)              # exact number of days in the 10-year span

# Temporal resolution: daily
dt_hours = 24.0                   # hours per step (daily)
dt = dt_hours                     # keep 'dt' name for continuity

# ------------------------------
# Grid
# ------------------------------
Nx, Ny = 256, 256
dx = dy = 1.0                     # spatial step (e.g., km or arbitrary units)

# ------------------------------
# Mean components
# ------------------------------
Gamma = 6.5 / 1000.0              # K per meter (environmental lapse rate)
beta0 = 290.0                     # baseline (K)

# Seasonal cycle (annual + optional 2nd harmonic)
A_y1 = 10.0   # K
A_y2 = 2.5    # K

# --- Center the seasonal peak on ~Aug 1 (middle of Jul–Aug) ---
days_in_year_ref = 365
f_peak = (date(2001, 8, 1) - date(2001, 1, 1)).days / days_in_year_ref  # 212/365

phi_y1 = np.pi/2 - 2*np.pi*f_peak
phi_y2 = np.pi/2 - 4*np.pi*f_peak   # aligns the 2nd harmonic on the same day
# ----------------------------------------------------------------

# (Keep diurnal for potential sub-daily runs; at daily sampling it's aliased → set to 0)
A_d, phi_d = 0.0, 0.0

# Synthetic elevation (smooth hill). Replace with real DEM if available.
x = np.arange(Nx) * dx; y = np.arange(Ny) * dy
X, Y = np.meshgrid(x, y, indexing='ij')
z = 500*np.exp(-((X-0.6*Nx*dx)**2 + (Y-0.4*Ny*dy)**2)/(2*(60*dx)**2))  # metres

# ------------------------------
# Spatio-temporal residual (AR(1) + advection)
# ------------------------------
u_x, u_y = 3.0, 0.0               # m/s (zonal, meridional). DAILY advection may be large; adjust if needed.
m_per_cell = 1000.0               # if dx=1 → 1 km/cell
shift_x = (u_x * dt*3600.0) / m_per_cell / dx   # pixels per time step
shift_y = (u_y * dt*3600.0) / m_per_cell / dy

tau_days = 7.0                    # AR(1) decorrelation time (days)
rho = np.exp(-1.0 / tau_days)     # persistence per daily step

# Matérn-like spectrum parameters for innovations
ell = 20.0                        # correlation length (cells)
nu = 1.5                          # smoothness
sigma_eta = 1.2                   # K, innovation std per step

# ------------------------------
# Spectral machinery
# ------------------------------
kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
ky = 2*np.pi*rfftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K = np.sqrt(KX**2 + KY**2) + 1e-12

# Matérn(ν,ℓ): S(k) ∝ (κ^2 + ||k||^2)^(-(ν + d/2)), κ = sqrt(2ν)/ℓ, d=2
kappa = np.sqrt(2*nu)/ell
S = (kappa**2 + K**2) ** (-(nu + 1.0))   # unnormalized, d=2 → ν + d/2 = ν + 1
S *= (sigma_eta**2) / np.mean(S)         # scale so innovations have target variance

rng = np.random.default_rng(42)

def sample_spatial_field():
    # rfftn/irfftn handles Hermitian symmetry for last axis
    W = (rng.normal(size=S.shape) + 1j*rng.normal(size=S.shape)) / np.sqrt(2.0)
    return irfftn(np.sqrt(S) * W, s=(Nx, Ny)).real

# ------------------------------
# Initialize residual field
# ------------------------------
R = sample_spatial_field()

# ------------------------------
# Simulation
# ------------------------------
sigma_eps = 0.2
T_series = np.empty((T_steps, Nx, Ny), dtype=np.float32)

# Precompute static mean part from elevation
mu_xy = beta0 - Gamma*z

# For seasonal phases, use exact day-of-year length each year to avoid drift
def doy_fraction(d):
    year = d.year
    is_leap = (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)
    days_in_year = 366 if is_leap else 365
    return (d - date(year, 1, 1)).days / days_in_year

for t, d in tqdm(enumerate(dates)):
    # Seasonal cycle (annual + second harmonic)
    f = doy_fraction(d)                 # in [0,1)
    mu_season = (
        A_y1 * np.sin(2*np.pi*f + phi_y1) +
        A_y2 * np.sin(4*np.pi*f + phi_y2)
    )
    # Optional diurnal (kept zero for daily sampling)
    mu_diurnal = A_d * np.sin(2*np.pi*(t*dt_hours/24.0) + phi_d)

    mu = mu_xy + mu_season + mu_diurnal

    # Advect previous residual (fractional-pixel shift)
    R_adv = nd_shift(R, shift=(shift_x, shift_y), order=1, mode='reflect')

    # New innovation
    eta = sample_spatial_field()

    # AR(1) update
    R = rho*R_adv + eta

    # Observation noise
    T_series[t] = (mu + R + rng.normal(0, sigma_eps, size=(Nx, Ny))).astype(np.float32)

# Result: 
# - T_series: array of shape [time (=len(dates)), Nx, Ny], in Kelvin
# - dates: list of datetime.date objects matching the first axis


# Example: T_series is your synthetic dataset [time, x, y].
# You can now write it to NetCDF, Zarr, or Parquet with x/y/time coords.
# --- Requirements ---
# pip install imageio matplotlib numpy

import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from matplotlib import cm, colors

# ============ CONFIG ============
out_gif = "temperature_daily.gif"
fps = 10                           # frames per second in the GIF
annotate = True                    # print the date on each frame
cmap = cm.get_cmap("RdYlBu_r")     # diverging, cold=blue, warm=red
dpi = 120                          # rendering resolution
# =================================

# Convert K -> °C
T_C = T_series - 273.15            # shape [T, Nx, Ny]

# Fix consistent color limits from robust percentiles across whole period
vmin = float(np.nanpercentile(T_C, 2.0))
vmax = float(np.nanpercentile(T_C, 98.0))
norm = colors.Normalize(vmin=vmin, vmax=vmax)

frames = []

# Optional: downsample for lighter GIFs (uncomment if needed)
# T_C = T_C[:, ::2, ::2]

for t in tqdm(range(T_C.shape[0])):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    im = ax.imshow(T_C[t], origin="lower", interpolation="nearest", norm=norm, cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (°C)")
    ax.set_xticks([]); ax.set_yticks([])
    if annotate:
        ax.set_title(dates[t].isoformat(), fontsize=10, loc="left")
    fig.tight_layout()

    # Render the Matplotlib figure to an RGB array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    frame = buf[..., :3]  # drop alpha

    frames.append(frame)
    plt.close(fig)


# Write animated GIF
# duration per frame (s) = 1/fps
iio.imwrite(out_gif, frames, duration=1.0/fps, loop=0)
print(f"Saved GIF: {out_gif}  | frames={len(frames)} | vmin={vmin:.2f}°C vmax={vmax:.2f}°C")


