from __future__ import annotations
import os
import numpy as np
import pandas as pd
from tsl.datasets.prototypes import DatetimeDataset
import matplotlib.pyplot as plt

# ---------------------------
# utilities
# ---------------------------

def _make_sinusoid_temperature(dates: pd.DatetimeIndex,
                               t_min: float = 5.0,
                               t_max: float = 30.0,
                               hottest_day: int = 200) -> np.ndarray:
    A = 0.5 * (t_max - t_min)
    mu = 0.5 * (t_max + t_min)
    phase = 2 * np.pi * ((dates.dayofyear.values - hottest_day) / 365.25)
    return mu + A * np.sin(phase)

def _make_nodes(n_nodes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=(n_nodes, 2))

def _build_adjacency(coords: np.ndarray, radius: float = 0.3, self_loops: bool = False):
    diffs = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    sigma2 = (radius / 2.0) ** 2
    A = np.exp(-dist**2 / (2 * sigma2)) * (dist <= radius)
    np.fill_diagonal(A, 1.0 if self_loops else 0.0)
    row_sum = A.sum(axis=1, keepdims=True)
    W = np.divide(A, row_sum, out=np.zeros_like(A), where=row_sum > 0)
    return dist, A, W

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# ---------------------------
# main dataset (modified)
# ---------------------------

class OstriniaSimpleSynth(DatetimeDataset):
    """
    Synthetic daily dataset with per-node thresholds and optional continuous (sigmoid) output.

    p_{i,t} = sigmoid(k * [ (T_{i,t} - T_thr_i)_+ + λ * (W (T_{·,t} - T_thr)_+ )_i ])
    Onset gating by smoothed T is kept; when continuous_target=True, target = p (sigmoid curve).
    """

    similarity_options = {'distance'}

    def __init__(self,
                 root: str | os.PathLike = "datasets",
                 years: list[int] = (2022, 2023),
                 freq: str | None = "D",
                 # graph
                 n_nodes: int = 25,
                 radius: float = 0.28,
                 seed: int = 7,
                 # temperature signal
                 t_min: float = 5.0,
                 t_max: float = 30.0,
                 hottest_day: int = 200,
                 temp_noise_sigma: float = 0.6,
                 # thresholds (NEW: per-node)
                 per_node_threshold: bool = True,
                 thr_mu: float = 25.0,        # mean threshold
                 thr_sigma: float = 1.5,      # std for per-node thresholds
                 # onset based on temperature threshold (still used as a gate)
                 onset_smooth_window: int = 5,   # days for MA before comparing to threshold
                 # probability shaping
                 spatial_coupling: float = 0.30, # λ
                 k_logit: float = 1.75,          # sigmoid slope
                 # output interface (NEW)
                 continuous_target: bool = True, # if True, target = p; else Bernoulli(p)
                 # legacy / misc interface
                 target: str = "nb_ostrinia",
                 input_zeros: bool = True,
                 smooth: bool = False,
                 full_normalization: bool = False,
                 add_second_target: bool = True,
                 delay: int = 14,
                 spatial_information: bool = True):

        # store
        self.root = str(root)
        self.target = target
        self.freq = freq
        self.smooth = smooth
        self.full_normalization = full_normalization
        self.add_second_target = add_second_target and (not continuous_target)  # disable for continuous targets
        self.delay = delay
        self.spatial_information = spatial_information

        self._years = list(years)
        self._n_nodes = int(n_nodes)
        self._radius = float(radius)
        self._seed = int(seed)
        self._tmin = float(t_min)
        self._tmax = float(t_max)
        self._hday = int(hottest_day)
        self._temp_noise_sigma = float(temp_noise_sigma)

        self._per_node_thr = bool(per_node_threshold)
        self._thr_mu = float(thr_mu)
        self._thr_sigma = float(thr_sigma)

        self._onset_win = int(onset_smooth_window)
        self._lambda = float(spatial_coupling)
        self._k = float(k_logit)
        self._continuous = bool(continuous_target)

        # build synthetic data
        df_long, dist, coords = self.load_raw()

        # pivot to wide (Date × node)
        df_long = df_long.set_index('Date')
        df_target = df_long.pivot(columns='node_index', values=self.target)

        if self.smooth:
            df_target = df_target.rolling(window=7, min_periods=7, center=True).mean()

        mask = (~df_target.isna()).astype('uint8').to_numpy()
        if input_zeros:
            df_target = df_target.fillna(0)

        # flags / covariates
        self.flags = {}
        extra_data = {}

        if self.add_second_target and 'increment_flag' in df_long.columns:
            inc = df_long.pivot(columns='node_index', values='increment_flag').fillna(0)
            self.flags['increment_flag'] = inc

        for col in ['temp', 'temp_clean', 'p', 'day_of_year', 'onset_active', 'thr_node']:
            if col in df_long.columns:
                extra_data[col] = df_long.pivot(columns='node_index', values=col).fillna(0)

        self.extra_data = extra_data

        super().__init__(target=df_target,
                         mask=mask,
                         freq=freq,
                         similarity_score="distance",
                         temporal_aggregation="nearest",
                         name="OstriniaSynth")

        if self.spatial_information:
            self.add_covariate('dist', dist, pattern='n n')
            self.add_covariate('coords', coords, pattern='n f')

        # plot the nodes specified in a list in the folder plots/synth
        nodes_to_plot = [0, 1, 2, 3, 4] if n_nodes >= 5 else list(range(n_nodes))

        # Plot each selected node
        for node in nodes_to_plot:
            if node in df_target.columns:
                plt.figure(figsize=(8, 4))
                plt.plot(df_target.index, df_target[node], label=node)
                plt.title(f"Signal from {node}")
                plt.xlabel("Date")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"plots/synth/{node}_signal.png", dpi=300)
                plt.close()

    # ------------------------------------------
    # synthesis with per-node thresholds
    # ------------------------------------------
    def load_raw(self):
        rng = np.random.default_rng(self._seed)
        coords = _make_nodes(self._n_nodes, self._seed)
        dist, A, W = _build_adjacency(coords, radius=self._radius, self_loops=False)

        # per-node thresholds
        if self._per_node_thr:
            thr_node = rng.normal(self._thr_mu, self._thr_sigma, size=self._n_nodes)
            thr_node = np.clip(thr_node, self._tmin, self._tmax)
        else:
            thr_node = np.full(self._n_nodes, self._thr_mu, dtype=float)

        records = []
        for y in self._years:
            dates = pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D")
            T_clean = _make_sinusoid_temperature(dates, self._tmin, self._tmax, self._hday)
            node_offsets = rng.normal(0.0, self._temp_noise_sigma, size=self._n_nodes)

            # per-node temporal series
            T_all = np.vstack([T_clean + node_offsets[i] for i in range(self._n_nodes)])  # (n_nodes, n_days)

            # smooth temperature for onset detection (per-node thresholds)
            T_smooth = pd.DataFrame(T_all.T, index=dates).rolling(
                window=self._onset_win, min_periods=1, center=True
            ).mean().to_numpy().T  # back to (n_nodes, n_days)

            # onset indicator per node/day (first crossing -> keep True afterwards)
            onset_active = np.zeros_like(T_all, dtype=bool)
            for i in range(self._n_nodes):
                above = T_smooth[i] >= thr_node[i]
                onset_active[i] = np.logical_or.accumulate(above)

            # probability after onset: sigmoid of temperature excess + spatial excess (per-node thr)
            temp_excess = np.clip(T_all - thr_node[:, None], a_min=0.0, a_max=None)  # (n_nodes, n_days)

            # spatial term each day
            spatial_term = np.empty_like(temp_excess)
            for t_idx in range(temp_excess.shape[1]):
                spatial_term[:, t_idx] = W @ temp_excess[:, t_idx]

            logits = self._k * (temp_excess + self._lambda * spatial_term)
            p_all = _sigmoid(logits) * onset_active  # gate before onset

            if self._continuous:
                Y = p_all  # continuous sigmoid-shaped target
            else:
                Y = rng.binomial(1, p_all)  # discrete draws

            # rows
            for t_idx, date in enumerate(dates):
                for i in range(self._n_nodes):
                    records.append({
                        "Date": date,
                        "year": y,
                        "day_of_year": date.dayofyear,
                        "node_index": i,
                        "x": coords[i, 0],
                        "y": coords[i, 1],
                        "thr_node": float(thr_node[i]),
                        "temp": float(T_all[i, t_idx]),
                        "temp_clean": float(T_clean[t_idx]),
                        "p": float(p_all[i, t_idx]),
                        "onset_active": int(onset_active[i, t_idx]),
                        "nb_ostrinia": float(Y[i, t_idx]) if self._continuous else int(Y[i, t_idx]),
                    })

        df = pd.DataFrame(records)
        if df.empty:
            raise RuntimeError("Synthetic generation produced an empty DataFrame.")

        # Optional second target only for discrete counts
        if self.add_second_target:
            df.sort_values(["node_index", "Date"], inplace=True)
            df["cum_year"] = (
                df.groupby(["node_index", df["Date"].dt.year])["nb_ostrinia"].cumsum()
            )
            df["increment_flag"] = (
                df.groupby(["node_index", df["Date"].dt.year])["cum_year"]
                  .diff()
                  .fillna(0)
                  .ne(0)
                  .astype(int)
            ).shift(-1, fill_value=0)
            df.drop(columns=["cum_year"], inplace=True)

        return df, dist, coords

    # ------------------------------------------
    # connectivity helper
    # ------------------------------------------
    def get_connectivity(self, layout, **kwargs):
        if layout == 'edge_index':
            from tsl.ops.connectivity import adj_to_edge_index
            dist = self._covariates['dist']['value']
            A = (dist > 0).astype(float)
            np.fill_diagonal(A, 0.0)
            return adj_to_edge_index(A)
        elif layout == 'distance':
            return self._covariates['dist']['value']
        else:
            raise ValueError(f"Unsupported layout: {layout}")

    def maybe_build(self):
        return
