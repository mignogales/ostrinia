from pathlib import Path
from datasets.peakweather import PeakWeatherTSL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Sequence, Tuple, Dict, Optional

from dataset_generator.functions import simulate_hatching_from_dd, degree_days_from_multiindex_wide, simulate_hatching_from_dd_v2

dataset = PeakWeatherTSL(
                 root="datasets",
                   input_zeros=True,
                    station_type="meteo_station",
                     freq='h',
                      target='temperature',
                        synth_data = "v3", 
                      )

adj = dataset.get_connectivity(threshold=20, include_self=True, layout='dist')
adj = 1 - adj/adj.max()  # convert to similarity
adj[adj < 0.9] = 0.0  # cap max weight to 1.0

# compute amount of non zero nodes aside from the diagonal
non_zero = np.sum(adj > 0, axis=1) - 1
print(f"Average number of connections per node (excluding self): {np.mean(non_zero):.2f} ± {np.std(non_zero):.2f}")
print(f"Max number of connections for a node (excluding self): {np.max(non_zero)}")
print(f"Min number of connections for a node (excluding self): {np.min(non_zero)}")


print(dataset)


res = degree_days_from_multiindex_wide(dataset.target, base=10.0, upper=30.0)

print(res.head())

def plot_nodes_dd(
    dd_wide: pd.DataFrame,
    nodes: Union[str, Sequence[str]] = "all",
    start=None,
    end=None,
    mode: str = "overlay",
    reset_annual: bool = True,
    save: bool = True,
    filename_prefix: str = "degree_days"
) -> Union[Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]], Dict[str, Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]]]:
    """
    Plot daily DD and cumulative DD for one or more nodes from the wide output produced by
    degree_days_from_multiindex_wide(...).

    Parameters
    ----------
    dd_wide : DataFrame
        Wide DataFrame with MultiIndex columns. Must include ('dd', node). ('dd_cum', node) is optional.
    nodes : list[str] | str
        List of node names, a single name, or "all".
    start, end : str | pandas.Timestamp | None
        Optional inclusive date range.
    mode : {'overlay','separate'}
        'overlay' -> single figure with all nodes; 'separate' -> one figure per node.
    reset_annual : bool
        If True, cumulative DD resets each calendar year (Jan 1). If False, it is a running total.
    save : bool
        If True, save the generated figure(s).
    filename_prefix : str
        Prefix for saved file names.

    Returns
    -------
    overlay -> (fig, (ax_dd, ax_cum))
    separate -> {node: (fig, (ax_dd, ax_cum))}
    """
    # --- Validate structure ---
    if not isinstance(dd_wide.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns with at least ('dd', node).")
    if "dd" not in dd_wide.columns.get_level_values(0):
        raise ValueError("Top-level columns must include 'dd'.")

    # --- Resolve nodes argument ---
    available_nodes = pd.Index(dd_wide["dd"].columns)
    if isinstance(nodes, str):
        sel_nodes = list(available_nodes) if nodes.lower() == "all" else [nodes]
    else:
        sel_nodes = list(dict.fromkeys(nodes))  # unique, keep order

    missing = [n for n in sel_nodes if n not in available_nodes]
    if missing:
        raise KeyError(f"Node(s) not found: {missing}. Examples: {list(available_nodes[:8])}")

    # --- Date slicing ---
    if start is not None or end is not None:
        start = pd.to_datetime(start) if start is not None else dd_wide.index.min()
        end   = pd.to_datetime(end)   if end   is not None else dd_wide.index.max()
        idx_slice = slice(start, end)
    else:
        idx_slice = slice(dd_wide.index.min(), dd_wide.index.max())

    # --- Helpers ---
    def _cum_by_setting(dd_series: pd.Series) -> pd.Series:
        if reset_annual:
            if not isinstance(dd_series.index, pd.DatetimeIndex):
                raise ValueError("DatetimeIndex required when reset_annual=True.")
            # Reset at each calendar year boundary
            return dd_series.groupby(dd_series.index.to_period("Y")).cumsum()
        else:
            return dd_series.cumsum()

    def _get_series(node_name: str):
        dd = dd_wide["dd"][node_name].loc[idx_slice]
        cum = _cum_by_setting(dd)
        return dd, cum

    # --- Plotting ---
    if mode not in {"overlay", "separate"}:
        raise ValueError("mode must be 'overlay' or 'separate'.")

    if mode == "overlay":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for n in sel_nodes:
            dd, cum = _get_series(n)
            dd.plot(ax=ax1, label=n)
            cum.plot(ax=ax2, label=n)

        suffix = "Annual reset" if reset_annual else "Running total"
        ax1.set_title(f"Daily Degree-Days — overlay")
        ax1.set_ylabel("DD (°C·day)")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(loc="upper left", ncol=2)

        ax2.set_title(f"Cumulative Degree-Days — overlay ({suffix})")
        ax2.set_ylabel("Cum. DD (°C·day)")
        ax2.set_xlabel("Date")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(loc="upper left", ncol=2)

        fig.tight_layout()
        if save:
            fname = f"{filename_prefix}__overlay_{len(sel_nodes)}nodes_{'annual' if reset_annual else 'running'}.png"
            plt.savefig(fname, dpi=150)
            print(f"Saved figure {fname}")
        return fig, (ax1, ax2)

    else:  # separate
        out: Dict[str, Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]] = {}
        for n in sel_nodes:
            dd, cum = _get_series(n)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

            dd.plot(ax=ax1)
            ax1.set_title(f"Daily Degree-Days — {n}")
            ax1.set_ylabel("DD (°C·day)")
            ax1.grid(True, linestyle="--", alpha=0.4)

            suffix = "Annual reset" if reset_annual else "Running total"
            cum.plot(ax=ax2)
            ax2.set_title(f"Cumulative Degree-Days — {n} ({suffix})")
            ax2.set_ylabel("Cum. DD (°C·day)")
            ax2.set_xlabel("Date")
            ax2.grid(True, linestyle="--", alpha=0.4)

            fig.tight_layout()
            if save:
                fname = f"{filename_prefix}__{n}_{'annual' if reset_annual else 'running'}.png"
                plt.savefig(fname, dpi=150)
                print(f"Saved figure {fname}")
            out[n] = (fig, (ax1, ax2))
        return out

plot_nodes_dd(res, nodes= res.columns.get_level_values('nodes')[:10], start=pd.Timestamp('2017-01-01',tz='UTC'), end=pd.Timestamp('2017-12-31',tz='UTC'), save=True)

# get station heights
stations_h = dataset.dataset.stations_table['station_height']
n_eggs = lambda h: 1000 * np.exp(-h / 2000)  # decay with 1km scale

# get station number of eggs per year using elevation and as n_eggs_year_i = f(station_height_i) + 0.1*n_eggs_year_i-1
n_eggs_year = np.zeros((stations_h.shape[0], 10))
for year in range(10):
    if year == 0:
        n_eggs_year[:, year] = n_eggs(stations_h)
    else:
        n_eggs_year[:, year] = n_eggs(stations_h) + n_eggs_year[:, year - 1] @ (adj * 0.035)

n_eggs_year = n_eggs_year[:, 1:]

# convert to dataframe, index is year, columns are station names
n_eggs_year_df = pd.DataFrame(n_eggs_year.T, columns=res.columns.get_level_values('nodes')[:len(res.columns)//2], index=range(2017, 2026))

print(n_eggs_year_df)

import numpy as np
import pandas as pd



# ---------------------------
# Example (commented usage):
# ---------------------------
# dd_df: DataFrame with daily DD, index=DatetimeIndex (daily), columns=['N1','N2',...]
# eggs_per_year = 500                                 # 500 eggs for every node-year
#   or eggs_per_year = pd.Series({2023: 600, 2024: 450})
#   or eggs_per_year = pd.DataFrame(..., index=years, columns=nodes)
#
# res = simulate_hatching_from_dd(
#     dd_df,
#     eggs_per_year=500,
#     mode="cum",
#     dd50=170.0,
#     k=0.06,
#     p_max=0.18,
#     lag_days=15,
#     random_state=42,
#     return_probs=True
# )
# hatched = res["hatched"]        # daily hatch counts
# survivors = res["survivors"]    # survivors after each day
# hazard = res["hazard"]          # daily hazard in [0,1]
# cumdd = res.get("cumdd", None)  # within-year cumulative DD if mode='cum'
#
# # Sanity check per (year,node): total hatched equals assigned eggs (unless year truncated)
# check = hatched.groupby(hatched.index.year).sum()

out = simulate_hatching_from_dd_v2(res['dd'],
                                    adjacency_matrix=adj,
                                    eggs_first_year=n_eggs(stations_h),
                                    mode="cum",
                                    dd50=700.0,
                                    k=0.20,
                                    p_max=0.1,
                                    dd50_daily=70.0,
                                    k_daily=0.07,
                                    p_max_daily=0.15,
                                    dd50_cum=700.0,
                                    k_cum=0.25,
                                    p_max_cum=0.05,
                                    lag_days=30,
                                    random_state=42,
                                    return_probs=True,
                                    mix_alpha=0.2,
                                    propagation_rate=1.0,
                                    dispersal_efficiency=0.4
                                )

print(out['hatched'].head())

def save_hatching_by_year(
    hatched: pd.DataFrame,
    node: str,
    out_path: str | Path,
    *,
    cumulative: bool = False,     # plot cumulative hatchings within each year
    smooth: int | None = 7,       # rolling window (days) per year; None disables
    freq: str | None = None,      # optional resample, e.g. "W" or "7D"
    drop_feb29: bool = True,      # remove Feb 29 for alignment
    title: str | None = None,
    dpi: int = 200,
    transparent: bool = False,
):
    """
    Save an overlay plot of daily (or cumulative) hatchings by year for a single node.

    Parameters
    ----------
    hatched : DataFrame
        DatetimeIndex (daily or resample-able); columns are nodes; values are daily hatch counts (int).
    node : str
        Column to plot.
    out_path : str | Path
        Output file path (extension determines format, e.g., .png/.pdf/.svg).
    """
    if node not in hatched.columns:
        raise KeyError(f"Node '{node}' not in hatched columns.")
    if not isinstance(hatched.index, pd.DatetimeIndex):
        raise ValueError("hatched index must be a DatetimeIndex.")

    s = hatched[node].sort_index()

    # Optional resampling (sum counts within bins)
    if freq is not None:
        s = s.resample(freq).sum()

    # Optional: drop Feb 29
    if drop_feb29:
        s = s[~((s.index.month == 2) & (s.index.day == 29))]

    # Build per-year dataframe
    df = pd.DataFrame({"hatch": s})
    df["year"] = df.index.year
    df["doy"]  = df.index.dayofyear

    # Daily vs cumulative within each year
    if cumulative:
        df["val"] = df.groupby("year")["hatch"].cumsum()
        y_label = "Cumulative hatched eggs"
    else:
        df["val"] = df["hatch"].astype(float)
        y_label = "Daily hatched eggs"

    # Optional smoothing within each year (rolling over day-of-year)
    if smooth is not None and smooth > 1:
        df["val"] = (
            df.groupby("year", group_keys=False)
              .apply(lambda g: g.set_index("doy")["val"]
                                .rolling(smooth, min_periods=1, center=True)
                                .mean()
                                .reindex(g["doy"].values))
              .values
        )

    # Align DOY across years
    pivot = df.pivot(index="doy", columns="year", values="val").sort_index()

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for y in pivot.columns:
        ax.plot(pivot.index, pivot[y], label=str(y), linewidth=1.7)

    ax.set_xlabel("Day of year")
    ax.set_ylabel(y_label)
    if title is None:
        ttl = f"{node} — {'Cumulative' if cumulative else 'Daily'} hatchings by year"
        if smooth and smooth > 1:
            ttl += f" (rolling {smooth}d)"
        ax.set_title(ttl)
    else:
        ax.set_title(title)
    ax.legend(title="Year", ncols=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.01)

    # Save
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", transparent=transparent)
    plt.close(fig)


    print(f"Saved plot to {out_path}")

save_hatching_by_year(out['hatched'], node=res.columns.get_level_values('nodes')[0], out_path=Path("hatching_example.png"), cumulative=False, smooth=7, freq=None, drop_feb29=True, title=None, dpi=200, transparent=False)

# compute the cumsum of the hatched dataframe
hatched_cumsum = out['hatched'].groupby(out['hatched'].index.year).cumsum()

# plot the cumsum of the hatched dataframe for the first 5 stations
# Get the first 3 years from the data
first_3_years = sorted(hatched_cumsum.index.year.unique())[:3]
mask = hatched_cumsum.index.year.isin(first_3_years)

# Plot the filtered data for the first 5 stations
hatched_cumsum.loc[mask, hatched_cumsum.columns[:5]].plot(figsize=(10, 6))
plt.title(f"Cumulative hatched eggs (first 5 stations, {first_3_years[0]}-{first_3_years[-1]})")
plt.xlabel("Date")
plt.ylabel("Cumulative hatched eggs")
plt.grid(True, alpha=0.3)
plt.savefig("cumulative_hatched_5_stations_3y.png", dpi=200, bbox_inches="tight")
plt.close()

# Plot the filtered data for the first 5 stations
hatched_cumsum.loc[:, hatched_cumsum.columns[:5]].plot(figsize=(10, 6))
plt.title(f"Cumulative hatched eggs (first 5 stations)")
plt.xlabel("Date")
plt.ylabel("Cumulative hatched eggs")
plt.grid(True, alpha=0.3)
plt.savefig("cumulative_hatched_5_stations.png", dpi=200, bbox_inches="tight")
plt.close()
