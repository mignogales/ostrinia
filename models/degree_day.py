"""Degree-Day (GDD) baseline for Ostrinia adult-flight forecasting.

Classical thermal-time baseline used in agricultural entomology and IPM systems
(UC IPM, NEWA). Cumulative Growing Degree Days are computed from daily mean
temperature with the simple-average method:

    DD_d  = max(0, T_mean_d - T_base)
    GDD_t = sum_{d in season(t)} DD_d

The accumulated GDD is then mapped to the target variable (adult-flight trap
counts) via one of three regressions selected in the YAML config:

    * linear   -> OLS on [1, GDD, GDD^2]
    * logistic -> 3-parameter sigmoid (classical sigmoidal emergence curve)
    * isotonic -> monotone non-decreasing step function

The mapping can be fitted per node or pooled across nodes (``pooled=True``).
Pooling is recommended when the spatial information is sparse or the nodes are
physically interchangeable (the standard IPM convention).

References
----------
Baskerville, G.L., Emin, P. (1969). Rapid estimation of heat accumulation from
maximum and minimum temperatures. Ecology 50(3): 514-517.

Got, B., Labatte, J.-M., Piry, S. (1996). European corn borer (Lepidoptera:
Pyralidae) development time model. Environmental Entomology 25(2): 310-320.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from scipy.optimize import curve_fit
from sklearn.isotonic import IsotonicRegression


# ---------------------------------------------------------------------------
# 1. GDD computation (simple-average method)
# ---------------------------------------------------------------------------

def _coerce_datetime_index(index: Any, *, name: str) -> pd.DatetimeIndex:
    """Return a timezone-naive DatetimeIndex for robust resampling/reindexing."""
    dt_index = pd.DatetimeIndex(pd.to_datetime(index, errors="raise"))
    if dt_index.tz is not None:
        dt_index = dt_index.tz_localize(None)
    if dt_index.hasnans:
        raise ValueError(f"`{name}` contains invalid timestamps")
    return dt_index


def compute_gdd(
    temperature: np.ndarray,
    timestamps: pd.DatetimeIndex,
    t_base: float = 10.0,
    t_upper: Optional[float] = 30.0,
    cutoff: Literal["horizontal", "vertical"] = "horizontal",
    biofix_doy: int = 1,
) -> np.ndarray:
    """Compute cumulative Growing Degree Days with annual reset at a biofix.

    Parameters
    ----------
    temperature : np.ndarray of shape (T, N)
        Temperature time series (degC), one column per node.
    timestamps : pd.DatetimeIndex of length T
        Timestamps aligned with the rows of ``temperature``.
    t_base : float
        Lower developmental threshold (degC). Default 10 (O. nubilalis).
    t_upper : float or None
        Upper developmental threshold (degC). ``None`` disables the upper cutoff.
    cutoff : {"horizontal", "vertical"}
        ``horizontal`` clips the daily mean to ``t_upper``; ``vertical`` zeros
        the daily contribution when the mean exceeds ``t_upper``.
    biofix_doy : int
        Day-of-year at which cumulative GDD is reset to 0 (1 = 1 January).

    Returns
    -------
    gdd : np.ndarray of shape (T, N)
        Cumulative GDD aligned with the input timestamps.
    """
    if temperature.ndim != 2:
        raise ValueError(f"`temperature` must be 2-D (T, N); got {temperature.shape}")
    timestamps = _coerce_datetime_index(timestamps, name="timestamps")
    if len(timestamps) != temperature.shape[0]:
        raise ValueError("Length of `timestamps` must equal temperature.shape[0]")

    # --- daily aggregation -------------------------------------------------
    df = pd.DataFrame(temperature, index=timestamps)
    daily_mean = df.resample("D").mean()
    daily_t = daily_mean.values

    if t_upper is not None:
        if cutoff == "horizontal":
            daily_t = np.minimum(daily_t, t_upper)
        elif cutoff == "vertical":
            daily_t = np.where(daily_t > t_upper, t_base, daily_t)
        else:
            raise ValueError(f"Unknown cutoff: {cutoff!r}")

    daily_dd = np.maximum(0.0, daily_t - t_base)

    # --- cumulative with annual reset -------------------------------------
    doy = daily_mean.index.dayofyear.values
    year = daily_mean.index.year.values
    season_key = np.where(doy < biofix_doy, year - 1, year)
    cum = np.zeros_like(daily_dd)
    for season in np.unique(season_key):
        m = season_key == season
        cum[m] = np.cumsum(daily_dd[m], axis=0)

    cum_df = pd.DataFrame(cum, index=daily_mean.index, columns=df.columns)

    # --- broadcast back to original frequency -----------------------------
    gdd = cum_df.reindex(timestamps, method="ffill").values
    gdd = np.nan_to_num(gdd, nan=0.0)
    return gdd.astype(np.float32)


def gdd_to_target_frame(
    gdd: np.ndarray,
    timestamps: pd.DatetimeIndex,
    target_index: pd.Index,
    columns: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """Align computed GDD to the target timeline used by the TSL dataset.

    ``compute_gdd`` returns values at the temperature sampling frequency. The
    forecasting dataset may be daily, or may carry string dates even when the
    temperature frame is already datetime-like. Aligning through a DataFrame
    keeps the covariate explicitly indexed before it is attached horizon-wise.
    """
    timestamps = _coerce_datetime_index(timestamps, name="timestamps")
    target_index = _coerce_datetime_index(target_index, name="target_index")
    if len(timestamps) != gdd.shape[0]:
        raise ValueError("Length of `timestamps` must equal gdd.shape[0]")

    gdd_df = pd.DataFrame(gdd, index=timestamps, columns=columns)
    if gdd_df.index.has_duplicates:
        gdd_df = gdd_df.groupby(level=0).last()

    aligned = gdd_df.reindex(target_index, method="ffill")
    return aligned.fillna(0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. GDD -> target mappings
# ---------------------------------------------------------------------------

def _logistic(g, L, k, g0):
    z = np.clip(-k * (g - g0), -50.0, 50.0)
    return L / (1.0 + np.exp(z))


@dataclass
class FittedMapping:
    """Container for a fitted GDD -> target mapping."""

    kind: str
    params: Dict[str, Any] = field(default_factory=dict)

    def predict(self, gdd: np.ndarray) -> np.ndarray:
        if self.kind == "linear":
            coef = self.params["coef"]
            X = np.stack([np.ones_like(gdd), gdd, gdd ** 2], axis=-1)
            return (X @ coef).astype(np.float32)
        if self.kind == "logistic":
            return _logistic(
                gdd, self.params["L"], self.params["k"], self.params["g0"]
            ).astype(np.float32)
        if self.kind == "isotonic":
            shape = gdd.shape
            return self.params["model"].predict(gdd.ravel()).reshape(shape).astype(np.float32)
        if self.kind == "constant":
            return np.full_like(gdd, self.params["value"], dtype=np.float32)
        raise ValueError(f"Unknown mapping kind: {self.kind}")

    def summary(self) -> Dict[str, Any]:
        """Loggable summary (sklearn objects are not serialised)."""
        if self.kind == "isotonic":
            return {"kind": self.kind, "params": "<IsotonicRegression>"}
        if self.kind == "linear":
            return {"kind": self.kind, "coef": self.params["coef"].tolist()}
        return {"kind": self.kind, **{k: float(v) for k, v in self.params.items()}}


def _fit_linear(g: np.ndarray, y: np.ndarray) -> FittedMapping:
    X = np.stack([np.ones_like(g), g, g ** 2], axis=-1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return FittedMapping(kind="linear", params={"coef": coef.astype(np.float64)})


def _fit_logistic(g: np.ndarray, y: np.ndarray) -> FittedMapping:
    y_max_raw = float(np.nanmax(y))
    y_max = y_max_raw if np.isfinite(y_max_raw) and y_max_raw > 0 else 1.0
    half = y > (y_max / 2.0)
    g_mid = float(np.median(g[half])) if half.any() else float(np.median(g))
    p0 = (y_max, 0.01, g_mid)
    try:
        popt, _ = curve_fit(
            _logistic,
            g,
            y,
            p0=p0,
            maxfev=10_000,
            bounds=(
                [0.0, 0.0, float(g.min())],
                [10.0 * y_max + 1e-6, 1.0, float(g.max()) + 1.0],
            ),
        )
        return FittedMapping(
            kind="logistic",
            params={"L": float(popt[0]), "k": float(popt[1]), "g0": float(popt[2])},
        )
    except (RuntimeError, ValueError):
        # Fall back to linear if non-linear fit fails
        return _fit_linear(g, y)


def _fit_isotonic(g: np.ndarray, y: np.ndarray) -> FittedMapping:
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(g, y)
    return FittedMapping(kind="isotonic", params={"model": iso})


_FITTERS = {
    "linear": _fit_linear,
    "logistic": _fit_logistic,
    "isotonic": _fit_isotonic,
}


def fit_degree_day(
    gdd_train: np.ndarray,
    y_train: np.ndarray,
    mapping: Literal["linear", "logistic", "isotonic"] = "linear",
    pooled: bool = False,
    mask: Optional[np.ndarray] = None,
) -> List[FittedMapping]:
    """Fit a per-node or pooled GDD -> target mapping.

    Parameters
    ----------
    gdd_train : np.ndarray of shape (T, N)
        Cumulative GDD over the training timeline.
    y_train : np.ndarray of shape (T, N)
        Target variable (adult-flight trap counts) over the training timeline.
    mapping : {"linear", "logistic", "isotonic"}
        Functional form of the mapping.
    pooled : bool
        If True, fit a single mapping using all (time, node) pairs concatenated.
        If False, fit one mapping per node.
    mask : np.ndarray of shape (T, N) or None
        Boolean validity mask; ``mask == 0`` entries are excluded from the fit.

    Returns
    -------
    fitted : list of FittedMapping, length N
        One mapping per node. When ``pooled=True`` the same fitted object is
        replicated across all nodes for downstream uniformity.
    """
    if gdd_train.shape != y_train.shape:
        raise ValueError(
            f"Shape mismatch: gdd {gdd_train.shape} vs y {y_train.shape}"
        )
    if mapping not in _FITTERS:
        raise ValueError(
            f"Unknown mapping {mapping!r}; choose from {list(_FITTERS)}"
        )
    fitter = _FITTERS[mapping]
    _, N = y_train.shape

    def _valid(g: np.ndarray, y: np.ndarray, m: Optional[np.ndarray]) -> np.ndarray:
        v = ~np.isnan(y) & ~np.isnan(g)
        if m is not None:
            v &= m.astype(bool)
        return v

    if pooled:
        g = gdd_train.reshape(-1).astype(np.float64)
        y = y_train.reshape(-1).astype(np.float64)
        m = mask.reshape(-1) if mask is not None else None
        v = _valid(g, y, m)
        if v.sum() < 5:
            mean = float(np.nanmean(y_train)) if np.isfinite(np.nanmean(y_train)) else 0.0
            fitted = FittedMapping(kind="constant", params={"value": mean})
        else:
            fitted = fitter(g[v], y[v])
        return [fitted for _ in range(N)]

    fitted_list: List[FittedMapping] = []
    for n in range(N):
        g = gdd_train[:, n].astype(np.float64)
        y = y_train[:, n].astype(np.float64)
        m = mask[:, n] if mask is not None else None
        v = _valid(g, y, m)
        if v.sum() < 5:
            mean = float(np.nanmean(y)) if np.isfinite(np.nanmean(y)) else 0.0
            fitted_list.append(FittedMapping(kind="constant", params={"value": mean}))
        else:
            fitted_list.append(fitter(g[v], y[v]))
    return fitted_list


# ---------------------------------------------------------------------------
# 3. Lightning wrapper (mirrors the ARIMAXWrapper interface used in train.py)
# ---------------------------------------------------------------------------

class DegreeDayWrapper(pl.LightningModule):
    """Lightning-compatible wrapper for the fitted Degree-Day baseline.

    The wrapper holds the per-node (or pooled) fitted mappings and exposes a
    ``test_step`` that reads horizon-aligned GDD from the batch (added in
    ``train.py`` via ``torch_dataset.add_covariate(name='gdd', ...,
    synch_mode='horizon', add_to_input_map=True)``) and applies the mapping.

    The constructor signature absorbs the unified kwargs from ``train.py`` and
    silently ignores those that are not relevant to a non-trainable baseline
    (optimizer, scheduler, loss, model_kwargs).
    """

    def __init__(
        self,
        n_nodes: int,
        model_kwargs: Optional[dict] = None,
        metrics: Optional[dict] = None,
        loss_fn=None,
        **_unused,
    ) -> None:
        super().__init__()
        mk = model_kwargs or {}
        self.n_nodes = int(n_nodes)
        self.horizon = int(mk.get("horizon", 1))

        self.fitted: Optional[List[FittedMapping]] = None
        self.test_metrics = torch.nn.ModuleDict(metrics or {})
        # Loss function used to populate the 'test_loss' callback metric
        # expected by Wandb_callback. May be a tsl Masked* metric or a dict
        # (double-target case); we only honour the regression branch here.
        self.loss_fn = loss_fn
        # Dummy parameter so .freeze() / .to(device) behave normally
        self.register_parameter("_anchor", torch.nn.Parameter(torch.zeros(1)))

    # ------------------------------------------------------------------ API
    def set_fitted(self, fitted: List[FittedMapping]) -> None:
        if len(fitted) != self.n_nodes:
            raise ValueError(
                f"Expected {self.n_nodes} fitted mappings, got {len(fitted)}"
            )
        self.fitted = fitted

    # ----------------------------------------------------------- helpers
    @staticmethod
    def _extract_gdd_from_batch(batch) -> torch.Tensor:
        """Locate the horizon-aligned GDD covariate in a TSL batch.

        Tries a few attribute paths to remain robust across TSL versions.
        """
        # Most common location after add_covariate(add_to_input_map=True)
        for attr in ("gdd",):
            if hasattr(batch.input, attr):
                return getattr(batch.input, attr)
            if hasattr(batch, attr):
                return getattr(batch, attr)
        if isinstance(getattr(batch, "input", None), dict) and "gdd" in batch.input:
            return batch.input["gdd"]
        raise RuntimeError(
            "Could not locate the 'gdd' covariate in the batch. Ensure "
            "add_covariate(name='gdd', synch_mode='horizon', "
            "add_to_input_map=True) was called on the SpatioTemporalDataset."
        )

    def _apply_mapping(self, gdd: torch.Tensor) -> torch.Tensor:
        """Apply per-node fitted mappings to a (B, H, N) tensor of GDD values."""
        if self.fitted is None:
            raise RuntimeError("DegreeDayWrapper has no fitted mappings.")
        g_np = gdd.detach().cpu().numpy()
        if g_np.ndim == 4 and g_np.shape[-1] == 1:
            g_np = g_np.squeeze(-1)
        if g_np.ndim != 3:
            raise ValueError(
                f"Expected GDD with shape (B, H, N); got {g_np.shape}"
            )
        B, H, N = g_np.shape
        if N != self.n_nodes:
            raise ValueError(
                f"Node mismatch: GDD has N={N} but wrapper has n_nodes={self.n_nodes}"
            )
        out = np.empty((B, H, N), dtype=np.float32)
        for n in range(N):
            out[:, :, n] = self.fitted[n].predict(g_np[:, :, n])
        return torch.from_numpy(out).unsqueeze(-1).to(gdd.device)

    # ---------------------------------------------------------- Lightning
    def forward(self, batch):
        gdd = self._extract_gdd_from_batch(batch)
        return self._apply_mapping(gdd)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """Same path as test_step, but returns predictions for save_predictions
        / trainer.predict() consumers. Returns a dict with both y_hat and the
        ground truth + mask so downstream code can compute residuals.
        """
        y_hat = self.forward(batch)
        y = batch.target.y if hasattr(batch, "target") else getattr(batch, "y", None)
        mask = getattr(batch, "mask", None)
        return {"y_hat": y_hat, "y": y, "mask": mask}

    def test_step(self, batch, batch_idx):
        y_hat = self.forward(batch)

        # Ground truth + mask (TSL batch conventions)
        y = batch.target.y if hasattr(batch, "target") else batch.y
        if hasattr(batch, "mask"):
            mask = batch.mask
        elif hasattr(batch, "transform") and "y" in getattr(batch, "transform", {}):
            mask = None
        else:
            mask = None

        # --- test_loss (expected by Wandb_callback's on_test_end) ----------
        if self.loss_fn is not None:
            loss_obj = (
                self.loss_fn["loss_regression"]
                if isinstance(self.loss_fn, dict)
                else self.loss_fn
            )
            try:
                test_loss = (
                    loss_obj(y_hat, y, mask) if mask is not None else loss_obj(y_hat, y)
                )
            except TypeError:
                # Some metric implementations require explicit kwarg names
                test_loss = loss_obj(y_hat, y)
            self.log("test_loss", test_loss, on_step=False, on_epoch=True,
                     batch_size=y.shape[0])

        for name, metric in self.test_metrics.items():
            metric.update(y_hat, y, mask) if mask is not None else metric.update(y_hat, y)
            self.log(f"test_{name}", metric, on_step=False, on_epoch=True)
        return {"y_hat": y_hat, "y": y}

    # Required for the .freeze() call in train.py
    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)

    def configure_optimizers(self):
        # No optimisation needed; return None and rely on test-only Trainer path.
        return None
