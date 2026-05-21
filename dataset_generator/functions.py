import pandas as pd
import numpy as np
from typing import Optional


def degree_days_from_multiindex_wide(
    df: pd.DataFrame,
    base: float = 10.0,
    upper: Optional[float] = 30.0,
    channel_label=0,
    reset_each_year: bool = False,
    start_date: Optional[str | pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Compute daily degree-days from a *wide* temperature DataFrame and keep it wide.

    Input
    -----
    - Index: DatetimeIndex (hourly or irregular sampling)
    - Columns: one per node OR a 2-level MultiIndex (node, channel) / (channel, node)
      with a single channel (default label=0). df[node_name] yields that node's series.

    Output
    ------
    - Wide DataFrame with DatetimeIndex at daily resolution.
      Columns are a MultiIndex: ('dd'|'dd_cum', node_name).
        - out['dd']     → daily degree-days per node (wide)
        - out['dd_cum'] → cumulative degree-days per node (wide)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex")

    # --- 1) Reduce to wide (nodes-only) if columns are MultiIndex with a single channel ---
    W = df.copy()
    if isinstance(W.columns, pd.MultiIndex):
        cols = W.columns
        if cols.nlevels != 2:
            raise ValueError("Expected 2-level columns: (node, channel) or (channel, node).")
        # Drop the level that is constant (e.g., channels=={0}); else select provided channel
        if cols.get_level_values(0).nunique() == 1:
            W = W.droplevel(0, axis=1)
        elif cols.get_level_values(1).nunique() == 1:
            W = W.droplevel(1, axis=1)
        else:
            # explicitly select the channel level wherever it sits
            try:
                W = W.xs(channel_label, level=0, axis=1, drop_level=True)
            except KeyError:
                W = W.xs(channel_label, level=1, axis=1, drop_level=True)

    # Ensure numeric and time-sorted; optionally trim start
    W = W.apply(pd.to_numeric, errors="coerce").sort_index()
    if start_date is not None:
        W = W[W.index >= pd.to_datetime(start_date)]
    if W.shape[0] < 2 or W.shape[1] == 0:
        # Return an empty wide frame with the expected column structure
        return pd.concat({"dd": W.iloc[0:0], "dd_cum": W.iloc[0:0]}, axis=1)

    # --- 2) Clip temperatures: T_eff = max( min(T, upper) - base, 0 ) ---
    T = W.to_numpy(dtype=float)
    if upper is not None:
        T = np.minimum(T, upper)
    T = np.maximum(T, base) - base  # ≥ 0

    # --- 3) Trapezoidal integration over time (attribute to the *ending* timestamp's day) ---
    idx = W.index
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    # ns since epoch → hours between samples
    epoch_ns = idx.view("int64")
    dt_hours = np.diff(epoch_ns) / 3.6e12  # (N-1,)

    T0, T1 = T[:-1, :], T[1:, :]
    trap = 0.5 * (T0 + T1) * dt_hours[:, None]  # °C·h
    valid = np.isfinite(T0) & np.isfinite(T1) & (dt_hours[:, None] > 0.0)
    trap = np.where(valid, trap, 0.0)
    dd_frac = trap / 24.0  # degree-days for each end-interval row

    # --- 4) Aggregate by calendar day using end-of-interval timestamps ---
    end_dates = pd.to_datetime(W.index[1:]).normalize()  # midnight of that day
    dd_df = pd.DataFrame(dd_frac, index=end_dates, columns=W.columns)
    dd_daily = dd_df.groupby(dd_df.index).sum()  # wide: date index, node columns

    # --- 5) Keep wide; compute cumulative per node (optionally reset per year) ---
    if reset_each_year:
        dd_cum = dd_daily.groupby(dd_daily.index.year).cumsum()
    else:
        dd_cum = dd_daily.cumsum()

    # Return a single wide frame with hierarchical columns: ('dd'|'dd_cum', node)
    out = pd.concat({"dd": dd_daily, "dd_cum": dd_cum}, axis=1)
    out.index.name = "datetime"
    # name the column levels
    out.columns.names = ["type", "nodes"]
    return out



def simulate_hatching_from_dd(
    dd_df: pd.DataFrame,
    eggs_per_year,
    *,
    mode: str = "daily",           # "cum", "daily", or "mixed"/"mixted"
    # Shared defaults (used unless specific *_daily/*_cum provided)
    dd50=100.0,                    # generic dd50
    k=0.05,                        # generic slope
    p_max=0.20,                    # generic cap on daily hatch probability
    # --- Per-driver overrides (optional) ---
    dd50_daily=None, k_daily=None, p_max_daily=None,
    dd50_cum=None,   k_cum=None,   p_max_cum=None,
    # ---
    lag_days: int = 0,
    clip_nonpositive: bool = True,
    random_state: int | None = 0,
    return_probs: bool = False,
    # Mixed-mode controls
    mix_rule: str = "mean",        # {"mean","noisy_or"}
    mix_alpha: float = 0.5         # weight for "mean": alpha*daily + (1-alpha)*cum
) -> dict:
    """
    dd50/k/p_max may be scalars or per-node Series/dict. Driver-specific overrides
    (dd50_daily/k_daily/p_max_daily and dd50_cum/k_cum/p_max_cum) are used if provided.
    Mixed mode combines *hazards* (not pre-sigmoid scores):
      - mean: p = α p_daily + (1-α) p_cum
      - noisy_or: p = 1 - (1 - p_daily)(1 - p_cum)
    Returns hatched/survivors and, if requested, the final per-day hazard.
    """
    # if not isinstance(dd_df.index, pd.DatetimeIndex):
    #     raise ValueError("dd_df index must be a daily DatetimeIndex.")
    dd_df = dd_df.sort_index()
    nodes = list(dd_df.columns)

    X = dd_df.copy()
    if clip_nonpositive:
        X = X.clip(lower=0)

    mode_norm = mode.lower()
    if mode_norm not in {"cum","daily","mixed","mixted"}:
        raise ValueError("mode must be 'cum', 'daily', or 'mixed'/'mixted'.")

    cumdd = None
    if mode_norm in {"cum","mixed","mixted"}:
        cumdd = X.groupby(X.index.year).cumsum()

    # --- param helpers ---
    def _to_series(p, name):
        # import numpy as np, pandas as pd
        if p is None:
            raise ValueError(f"Parameter '{name}' unexpectedly None after resolution.")
        if np.isscalar(p):
            return pd.Series({n: float(p) for n in nodes})
        if isinstance(p, (pd.Series, dict)):
            s = pd.Series(p, dtype=float)
            missing = set(nodes) - set(s.index)
            if missing:
                raise ValueError(f"Parameter '{name}' missing nodes: {sorted(missing)}")
            return s.loc[nodes].astype(float)
        raise ValueError(f"Parameter '{name}' must be scalar or Series/dict keyed by node.")

    def _resolve(override, base, name):
        return _to_series(override if override is not None else base, name)

    # Resolve per-driver params (fallback to shared)
    dd50_d = _resolve(dd50_daily, dd50, "dd50_daily")
    k_d    = _resolve(k_daily,    k,    "k_daily")
    pmax_d = _resolve(p_max_daily,p_max,"p_max_daily")

    dd50_c = _resolve(dd50_cum,   dd50, "dd50_cum")
    k_c    = _resolve(k_cum,      k,    "k_cum")
    pmax_c = _resolve(p_max_cum,  p_max,"p_max_cum")

    # Sigmoid per driver
    def _sigmoid(df, dd50_s, k_s):
        Z = (df - dd50_s.values) * k_s.values
        # numerically stable logistic
        return 1.0 / (1.0 + np.exp(-Z))

    # import numpy as np, pandas as pd

    if mode_norm == "daily":
        s_d = _sigmoid(X, dd50_d, k_d)
        hazard = s_d.multiply(pmax_d.values, axis=1)

    elif mode_norm == "cum":
        s_c = _sigmoid(cumdd, dd50_c, k_c)
        hazard = s_c.multiply(pmax_c.values, axis=1)

    else:  # mixed/mixted
        if mix_rule not in {"mean","noisy_or"}:
            raise ValueError("mix_rule must be 'mean' or 'noisy_or'.")
        if mix_rule == "mean" and not (0.0 <= mix_alpha <= 1.0):
            raise ValueError("mix_alpha must be in [0,1].")

        s_d = _sigmoid(X, dd50_d, k_d)
        s_c = _sigmoid(cumdd, dd50_c, k_c)
        p_d = s_d.multiply(pmax_d.values, axis=1)
        p_c = s_c.multiply(pmax_c.values, axis=1)

        if mix_rule == "mean":
            hazard = p_d * mix_alpha + p_c * (1.0 - mix_alpha)
        else:  # noisy_or
            hazard = 1.0 - (1.0 - p_d) * (1.0 - p_c)

        hazard = hazard.clip(lower=0.0, upper=1.0)

    # Enforce lag
    if lag_days > 0:
        mask = pd.Series(dd_df.index.dayofyear <= lag_days, index=dd_df.index)
        hazard.loc[mask, :] = 0.0

    # Eggs per (year, node)
    years = dd_df.index.year.unique()

    def _eggs_for_year(y):
        if isinstance(eggs_per_year, (int, np.integer, float)):
            return pd.Series({n: int(eggs_per_year) for n in nodes})
        if isinstance(eggs_per_year, pd.Series):
            if y not in eggs_per_year.index:
                raise ValueError(f"Year {y} missing in eggs_per_year Series.")
            val = int(eggs_per_year.loc[y]); return pd.Series({n: val for n in nodes})
        if isinstance(eggs_per_year, pd.DataFrame):
            if y not in eggs_per_year.index:
                raise ValueError(f"Year {y} missing in eggs_per_year DataFrame.")
            row = eggs_per_year.loc[y]
            missing = set(nodes) - set(row.index)
            if missing:
                raise ValueError(f"eggs_per_year DataFrame missing nodes: {sorted(missing)}")
            return row.loc[nodes].astype(int)
        raise ValueError("eggs_per_year must be int, Series(index=years), or DataFrame(years x nodes).")

    rng = np.random.default_rng(random_state)
    hatched_out, survivors_out = [], []

    for y in years:
        idx_y = dd_df.index[dd_df.index.year == y]
        H = pd.DataFrame(0, index=idx_y, columns=nodes, dtype=int)
        S = pd.DataFrame(0, index=idx_y, columns=nodes, dtype=int)

        survivors = _eggs_for_year(y).astype(int).values
        for t, day in enumerate(idx_y):
            p = np.clip(hazard.loc[day].values, 0.0, 1.0)
            hatch_t = rng.binomial(survivors, p)
            survivors = survivors - hatch_t
            H.iloc[t, :] = hatch_t
            S.iloc[t, :] = survivors

        hatched_out.append(H); survivors_out.append(S)

    hatched_df = pd.concat(hatched_out, axis=0)
    survivors_df = pd.concat(survivors_out, axis=0)

    out = {"hatched": hatched_df, "survivors": survivors_df}
    if return_probs:
        out["hazard"] = hazard
    if cumdd is not None:
        out["cumdd"] = cumdd

    # add cum hatched reset each year
    out["hatched_cum"] = hatched_df.groupby(hatched_df.index.year).cumsum()
    
    return out

def simulate_hatching_from_dd_v2(
    dd_df: pd.DataFrame,
    eggs_first_year,  # Changed: only first year eggs
    adjacency_matrix: np.ndarray,  # Added: spatial connectivity
    *,
    mode: str = "daily",           
    # Shared defaults
    dd50=100.0,                    
    k=0.05,                        
    p_max=0.20,                    
    # Per-driver overrides
    dd50_daily=None, k_daily=None, p_max_daily=None,
    dd50_cum=None,   k_cum=None,   p_max_cum=None,
    # Other parameters
    lag_days: int = 0,
    clip_nonpositive: bool = True,
    random_state: int | None = 0,
    return_probs: bool = False,
    # Mixed-mode controls
    mix_rule: str = "mean",        
    mix_alpha: float = 0.5,
    # New parameters for egg propagation
    propagation_rate: float = 0.5,  # local reproduction rate (will be normalized with dispersal_efficiency to sum to 1)
    dispersal_efficiency: float = 0.5,  # spatial dispersal rate (will be normalized with propagation_rate to sum to 1)
    daily_dispersal: bool = True  # whether to apply dispersal daily or only between seasons
) -> dict:
    """
    Enhanced hatching simulation with spatial connectivity and temporal dynamics.
    
    Parameters:
    -----------
    dd_df : pd.DataFrame
        Degree-day data with DatetimeIndex and node columns
    eggs_first_year : int, pd.Series, or dict
        Initial egg counts for first year only
    adjacency_matrix : np.ndarray
        Square matrix (nodes x nodes) defining spatial connectivity.
        Values represent connection strength (0 = no connection, 1 = full connection).
        Matrix order must match the column order in dd_df.
    propagation_rate : float
        Local reproduction rate (normalized with dispersal_efficiency to sum to 1)
    dispersal_efficiency : float  
        Spatial dispersal rate (normalized with propagation_rate to sum to 1)
    daily_dispersal : bool
        If True, apply dispersal daily during the season. If False, only between seasons.
        
    Returns:
    --------
    dict with keys: 'hatched', 'survivors', 'eggs_per_year', and optionally 'hazard', 'cumdd'
    """
    
    # Validate adjacency matrix
    if not isinstance(adjacency_matrix, np.ndarray):
        raise ValueError("adjacency_matrix must be a numpy ndarray")
    
    nodes = list(dd_df.columns)
    n_nodes = len(nodes)
    
    if adjacency_matrix.shape != (n_nodes, n_nodes):
        raise ValueError(f"adjacency_matrix must be square with shape ({n_nodes}, {n_nodes}) to match dd_df columns")
    
    adj_matrix = adjacency_matrix.copy()
    
    # Validate and normalize adjacency matrix
    if np.any(adj_matrix < 0):
        raise ValueError("Adjacency matrix cannot contain negative values")
    
    # Row-normalize adjacency matrix so each row sums to 1 (or 0 for isolated nodes)
    row_sums = np.sum(adj_matrix, axis=1, keepdims=True)
    # Avoid division by zero for isolated nodes (rows with all zeros)
    adj_matrix = np.divide(adj_matrix, row_sums, out=np.zeros_like(adj_matrix), where=row_sums!=0)
    
    # Normalize propagation parameters to sum to 1
    param_sum = propagation_rate + dispersal_efficiency
    if param_sum <= 0:
        raise ValueError("propagation_rate + dispersal_efficiency must be > 0")
    
    local_reproduction_rate = propagation_rate / param_sum
    spatial_dispersal_rate = dispersal_efficiency / param_sum
    
    dd_df = dd_df.sort_index()

    X = dd_df.copy()
    if clip_nonpositive:
        X = X.clip(lower=0)

    mode_norm = mode.lower()
    if mode_norm not in {"cum","daily","mixed","mixted"}:
        raise ValueError("mode must be 'cum', 'daily', or 'mixed'/'mixted'.")

    cumdd = None
    if mode_norm in {"cum","mixed","mixted"}:
        cumdd = X.groupby(X.index.year).cumsum()

    # Parameter resolution functions (unchanged)
    def _to_series(p, name):
        # import numpy as np, pandas as pd
        if p is None:
            raise ValueError(f"Parameter '{name}' unexpectedly None after resolution.")
        if np.isscalar(p):
            return pd.Series({n: float(p) for n in nodes})
        if isinstance(p, (pd.Series, dict)):
            s = pd.Series(p, dtype=float)
            missing = set(nodes) - set(s.index)
            if missing:
                raise ValueError(f"Parameter '{name}' missing nodes: {sorted(missing)}")
            return s.loc[nodes].astype(float)
        raise ValueError(f"Parameter '{name}' must be scalar or Series/dict keyed by node.")

    def _resolve(override, base, name):
        return _to_series(override if override is not None else base, name)

    # Resolve parameters
    dd50_d = _resolve(dd50_daily, dd50, "dd50_daily")
    k_d    = _resolve(k_daily,    k,    "k_daily")
    pmax_d = _resolve(p_max_daily,p_max,"p_max_daily")

    dd50_c = _resolve(dd50_cum,   dd50, "dd50_cum")
    k_c    = _resolve(k_cum,      k,    "k_cum")
    pmax_c = _resolve(p_max_cum,  p_max,"p_max_cum")

    # Sigmoid function (unchanged)
    def _sigmoid(df, dd50_s, k_s):
        Z = (df - dd50_s.values) * k_s.values
        return 1.0 / (1.0 + np.exp(-Z))

    # import numpy as np, pandas as pd

    # Calculate hazard based on mode (unchanged logic)
    if mode_norm == "daily":
        s_d = _sigmoid(X, dd50_d, k_d)
        hazard = s_d.multiply(pmax_d.values, axis=1)
    elif mode_norm == "cum":
        s_c = _sigmoid(cumdd, dd50_c, k_c)
        hazard = s_c.multiply(pmax_c.values, axis=1)
    else:  # mixed/mixted
        if mix_rule not in {"mean","noisy_or"}:
            raise ValueError("mix_rule must be 'mean' or 'noisy_or'.")
        if mix_rule == "mean" and not (0.0 <= mix_alpha <= 1.0):
            raise ValueError("mix_alpha must be in [0,1].")

        s_d = _sigmoid(X, dd50_d, k_d)
        s_c = _sigmoid(cumdd, dd50_c, k_c)
        p_d = s_d.multiply(pmax_d.values, axis=1)
        p_c = s_c.multiply(pmax_c.values, axis=1)

        if mix_rule == "mean":
            hazard = p_d * mix_alpha + p_c * (1.0 - mix_alpha)
        else:  # noisy_or
            hazard = 1.0 - (1.0 - p_d) * (1.0 - p_c)

        hazard = hazard.clip(lower=0.0, upper=1.0)

    # Apply lag
    if lag_days > 0:
        mask = pd.Series(dd_df.index.dayofyear <= lag_days, index=dd_df.index)
        hazard.loc[mask, :] = 0.0

    # Initialize first year eggs
    def _process_eggs_input(eggs_input):
        if isinstance(eggs_input, (int, np.integer, float)):
            return pd.Series({n: int(eggs_input) for n in nodes})
        if isinstance(eggs_input, pd.Series):
            missing = set(nodes) - set(eggs_input.index)
            if missing:
                raise ValueError(f"eggs_first_year Series missing nodes: {sorted(missing)}")
            return eggs_input.loc[nodes].astype(int)
        if isinstance(eggs_input, dict):
            s = pd.Series(eggs_input, dtype=int)
            missing = set(nodes) - set(s.index)
            if missing:
                raise ValueError(f"eggs_first_year dict missing nodes: {sorted(missing)}")
            return s.loc[nodes].astype(int)
        raise ValueError("eggs_first_year must be int, Series, or dict keyed by node.")

    # Function to calculate next year's eggs based on previous year's hatching
    def _calculate_next_year_eggs(total_hatched_prev_year):
        """
        Calculate eggs for next year with normalized rates to prevent population explosion.
        Total eggs = local reproduction + spatial dispersal, where rates sum to 1.
        """
        # Local reproduction: fraction of hatched individuals reproduce locally
        local_eggs = (total_hatched_prev_year * local_reproduction_rate).astype(int)
        
        # Spatial dispersal: fraction of hatched individuals disperse to connected nodes
        # Each node receives eggs from all nodes (including itself) based on normalized adjacency matrix
        dispersed_eggs = np.zeros(len(nodes))
        
        # Matrix multiplication: adj_matrix[i,j] = fraction of node j's dispersing individuals going to node i
        hatched_array = total_hatched_prev_year.values * spatial_dispersal_rate
        
        # Each row i in adj_matrix represents where node i receives dispersers from
        for i in range(len(nodes)):
            dispersed_eggs[i] = np.sum(hatched_array * adj_matrix[i, :])
        
        total_eggs = local_eggs.values + dispersed_eggs.astype(int)
        return pd.Series(total_eggs, index=nodes, dtype=int)

    def _apply_daily_dispersal(survivors):
        """
        Apply daily dispersal to survivors using the normalized adjacency matrix.
        
        Parameters:
        -----------
        survivors : np.ndarray
            Current survivor counts for each node
            
        Returns:
        --------
        np.ndarray
            Redistributed survivor counts after dispersal
        """
        # Apply spatial dispersal using adjacency matrix
        # Each row i in adj_matrix represents where node i receives dispersers from
        dispersed_survivors = np.zeros(len(survivors))
        
        for i in range(len(survivors)):
            # Node i receives survivors from all nodes (including itself) 
            # based on normalized adjacency matrix weights
            dispersed_survivors[i] = np.sum(survivors * adj_matrix[i, :])
        
        return dispersed_survivors.astype(int)

    years = dd_df.index.year.unique()
    rng = np.random.default_rng(random_state)
    
    hatched_out, still_not_hatched_out = [], []
    eggs_record = {}  # Track eggs per year for output
    
    # Process first year
    current_eggs = _process_eggs_input(eggs_first_year)
    eggs_record[years[0]] = current_eggs.copy()
    
    for year_idx, y in enumerate(years):
        idx_y = dd_df.index[dd_df.index.year == y]
        # already hatched (H) and still haven't hatched (S)
        H = pd.DataFrame(0, index=idx_y, columns=nodes, dtype=int)
        S = pd.DataFrame(0, index=idx_y, columns=nodes, dtype=int)

        still_not_hatched = current_eggs.astype(int).values
        
        for t, day in enumerate(idx_y):
            p = np.clip(hazard.loc[day].values, 0.0, 1.0)
            hatch_t = rng.binomial(still_not_hatched, p)
            still_not_hatched = still_not_hatched - hatch_t
            
            # Apply daily dispersal to hatched individuals
            if daily_dispersal and t < len(idx_y) - 1:  # Don't disperse on last day
                hatch_t = _apply_daily_dispersal(hatch_t)
            
            H.iloc[t, :] = hatch_t
            S.iloc[t, :] = still_not_hatched

        hatched_out.append(H)
        still_not_hatched_out.append(S)
        
        # Calculate eggs for next year based on final survivors (if not the last year)
        if year_idx < len(years) - 1:
            hatched_end = current_eggs - still_not_hatched  # Last day survivors for each node
            current_eggs = _calculate_next_year_eggs(hatched_end)
            eggs_record[years[year_idx + 1]] = current_eggs.copy()

    hatched_df = pd.concat(hatched_out, axis=0)
    survivors_df = pd.concat(still_not_hatched_out, axis=0)

    out = {
        "hatched": hatched_df, 
        "survivors": survivors_df,
        "eggs_per_year": pd.DataFrame.from_dict(eggs_record, orient='index')
    }
    
    if return_probs:
        out["hazard"] = hazard
    if cumdd is not None:
        out["cumdd"] = cumdd

    out["hatched_cum"] = hatched_df.groupby(hatched_df.index.year).cumsum()
    
    return out


def simulate_hatching_from_dd_v3(
    dd_df: pd.DataFrame,
    eggs_first_year,
    adjacency_matrix: np.ndarray,
    *,
    mode: str = "daily",
    # Shared defaults
    dd50=100.0,
    k=0.05,
    p_max=0.20,
    # Per-driver overrides
    dd50_daily=None, k_daily=None, p_max_daily=None,
    dd50_cum=None, k_cum=None, p_max_cum=None,
    # Other parameters
    lag_days: int = 0,
    clip_nonpositive: bool = True,
    random_state: int | None = 0,
    return_probs: bool = False,
    # Mixed-mode controls
    mix_rule: str = "mean",
    mix_alpha: float = 0.5,
    # Spatial dispersal parameters
    dispersal_rate: float = 0.5,  # Daily dispersal rate [0,1]
    propagation_rate: float = 1,  # Local reproduction for next season
    dispersal_efficiency: float = 0.5,  # Spatial dispersal for next season
    lag_dispersal_days: int = 1,  # Number of days before to use for dispersal
    soil_quality: Optional[pd.Series | dict] = None  # Not implemented yet
) -> dict:
    """
    Enhanced hatching simulation with *delayed* daily spatial dispersal.

    Change:
    -------
    - Daily dispersal now uses hatches from `lag_dispersal_days` before (default = 1).
      i.e., today's arrivals = today's hatches + (A_off @ hatches[t - lag_dispersal_days]) * dispersal_rate
    """

    # Validate dispersal_rate
    if not 0 <= dispersal_rate <= 1:
        raise ValueError("dispersal_rate must be in [0, 1]")

    if lag_dispersal_days < 1:
        raise ValueError("lag_dispersal_days must be >= 1")

    # Validate adjacency matrix
    if not isinstance(adjacency_matrix, np.ndarray):
        raise ValueError("adjacency_matrix must be a numpy ndarray")

    nodes = list(dd_df.columns)
    n_nodes = len(nodes)

    if adjacency_matrix.shape != (n_nodes, n_nodes):
        raise ValueError(
            f"adjacency_matrix must be square with shape ({n_nodes}, {n_nodes}) "
            f"to match dd_df columns"
        )

    A = adjacency_matrix.astype(float).copy()
    if np.any(A < 0):
        raise ValueError("Adjacency matrix cannot contain negative values")

    # Row-normalize full adjacency
    row_sums = A.sum(axis=1, keepdims=True)
    A = np.divide(A, row_sums, out=np.zeros_like(A), where=row_sums != 0)

    # Build off-diagonal normalized matrix (A - I)
    I = np.eye(n_nodes, dtype=float)
    A_off = A.copy()
    np.fill_diagonal(A_off, 0.0)
    row_sums_off = A_off.sum(axis=1, keepdims=True)
    A_off = np.divide(A_off, row_sums_off, out=np.zeros_like(A_off), where=row_sums_off != 0)

    # Normalize season-to-season propagation
    param_sum = propagation_rate + dispersal_efficiency
    if param_sum <= 0:
        raise ValueError("propagation_rate + dispersal_efficiency must be > 0")

    local_reproduction_rate = propagation_rate / param_sum
    spatial_dispersal_rate = dispersal_efficiency / param_sum

    dd_df = dd_df.sort_index()
    X = dd_df.copy()
    if clip_nonpositive:
        X = X.clip(lower=0)

    mode_norm = mode.lower()
    if mode_norm not in {"cum", "daily", "mixed"}:
        raise ValueError("mode must be 'cum', 'daily', or 'mixed'.")

    cumdd = None
    if mode_norm in {"cum", "mixed", "mixted"}:
        cumdd = X.groupby(X.index.year).cumsum()

    def _to_series(p, name):
        if p is None:
            raise ValueError(f"Parameter '{name}' unexpectedly None after resolution.")
        if np.isscalar(p):
            return pd.Series({n: float(p) for n in nodes})
        if isinstance(p, (pd.Series, dict)):
            s = pd.Series(p, dtype=float)
            missing = set(nodes) - set(s.index)
            if missing:
                raise ValueError(f"Parameter '{name}' missing nodes: {sorted(missing)}")
            return s.loc[nodes].astype(float)
        raise ValueError(f"Parameter '{name}' must be scalar or Series/dict keyed by node.")

    def _resolve(override, base, name):
        return _to_series(override if override is not None else base, name)

    dd50_d = _resolve(dd50_daily, dd50, "dd50_daily")
    k_d = _resolve(k_daily, k, "k_daily")
    pmax_d = _resolve(p_max_daily, p_max, "p_max_daily")

    dd50_c = _resolve(dd50_cum, dd50, "dd50_cum")
    k_c = _resolve(k_cum, k, "k_cum")
    pmax_c = _resolve(p_max_cum, p_max, "p_max_cum")

    def _sigmoid(df, dd50_s, k_s):
        Z = (df - dd50_s.values) * k_s.values
        return 1.0 / (1.0 + np.exp(-Z))

    # Compute hazard (same as before)
    if mode_norm == "daily":
        s_d = _sigmoid(X, dd50_d, k_d)
        hazard = s_d.multiply(pmax_d.values, axis=1)
    elif mode_norm == "cum":
        s_c = _sigmoid(cumdd, dd50_c, k_c)
        hazard = s_c.multiply(pmax_c.values, axis=1)
    else:  # mixed
        if mix_rule not in {"mean", "noisy_or"}:
            raise ValueError("mix_rule must be 'mean' or 'noisy_or'.")
        s_d = _sigmoid(X, dd50_d, k_d)
        s_c = _sigmoid(cumdd, dd50_c, k_c)
        p_d = s_d.multiply(pmax_d.values, axis=1)
        p_c = s_c.multiply(pmax_c.values, axis=1)
        hazard = p_d * mix_alpha + p_c * (1.0 - mix_alpha) if mix_rule == "mean" else 1.0 - (1.0 - p_d) * (1.0 - p_c)
        hazard = hazard.clip(0, 1)

    # use soil quality if provided to affect hazard
    if soil_quality is not None:
        hazard = hazard.multiply(soil_quality, axis=1)

    if lag_days > 0:
        mask = pd.Series(dd_df.index.dayofyear <= lag_days, index=dd_df.index)
        hazard.loc[mask, :] = 0.0

    def _process_eggs_input(eggs_input):
        if isinstance(eggs_input, (int, np.integer, float)):
            return pd.Series({n: int(eggs_input) for n in nodes})
        if isinstance(eggs_input, pd.Series):
            return eggs_input.loc[nodes].astype(int)
        if isinstance(eggs_input, dict):
            return pd.Series(eggs_input, dtype=int).loc[nodes]
        raise ValueError("eggs_first_year must be int, Series, or dict keyed by node.")

    years = dd_df.index.year.unique()
    rng = np.random.default_rng(random_state)

    hatched_out, still_not_hatched_out = [], []
    eggs_record = {}

    current_eggs = _process_eggs_input(eggs_first_year)
    eggs_record[years[0]] = current_eggs.copy()

    for year_idx, y in enumerate(years):
        idx_y = dd_df.index[dd_df.index.year == y]
        H = pd.DataFrame(0, index=idx_y, columns=nodes, dtype=int)
        S = pd.DataFrame(0, index=idx_y, columns=nodes, dtype=int)

        still_not_hatched = current_eggs.astype(int).values
        hatch_history = []  # to store previous days’ hatches

        for t, day in enumerate(idx_y):
            p = np.clip(hazard.loc[day].values, 0.0, 1.0)
            hatch_t = rng.binomial(still_not_hatched, p)
            still_not_hatched -= hatch_t

            # add gaussian noise
            # noise = np.random.normal(0, 0.1, size=hatch_t.shape) * hatch_t * 0.5
            # hatch_t = np.clip(hatch_t + noise, 0, None).astype(int)

            hatch_history.append(hatch_t.astype(float))

            # apply random movement of insects 
            if dispersal_rate > 0 and t >= lag_dispersal_days:
                # generate random noise in those nodes that had hatchings lag_dispersal_days ago, whose variance is proportional to hatchings
                past_hatched = hatch_history[t - lag_dispersal_days]
                # apply dispersal
                noise = rng.normal(0, 1, size=past_hatched.shape) * past_hatched * dispersal_rate
                redistributed = A_off @ noise
            else:
                redistributed = np.zeros(n_nodes, dtype=float)

            hatch_effective = hatch_t.astype(float) + redistributed

            H.iloc[t, :] = hatch_effective.astype(int)
            S.iloc[t, :] = still_not_hatched

        hatched_out.append(H)
        still_not_hatched_out.append(S)
        if year_idx < len(years) - 1:
            eggs_record[years[year_idx + 1]] = current_eggs.copy()

    hatched_df = pd.concat(hatched_out, axis=0)
    survivors_df = pd.concat(still_not_hatched_out, axis=0)

    out = {
        "hatched": hatched_df,
        "survivors": survivors_df,
        "eggs_per_year": pd.DataFrame.from_dict(eggs_record, orient='index'),
        "hatched_cum": hatched_df.groupby(hatched_df.index.year).cumsum(),
    }
    if return_probs:
        out["hazard"] = hazard
    if cumdd is not None:
        out["cumdd"] = cumdd
    return out




# def simulate_hatching_from_dd_v3(
#     dd_df: pd.DataFrame,
#     eggs_first_year,
#     adjacency_matrix: np.ndarray,
#     *,
#     mode: str = "daily",
#     # Shared defaults
#     dd50=100.0,
#     k=0.05,
#     p_max=0.20,
#     # Per-driver overrides
#     dd50_daily=None, k_daily=None, p_max_daily=None,
#     dd50_cum=None, k_cum=None, p_max_cum=None,
#     # Other parameters
#     lag_days: int = 0,
#     clip_nonpositive: bool = True,
#     random_state: int | None = 0,
#     return_probs: bool = False,
#     # Mixed-mode controls
#     mix_rule: str = "mean",
#     mix_alpha: float = 0.5,
#     # Spatial dispersal parameters
#     dispersal_rate: float = 0.5,  # Daily dispersal rate [0,1]
#     propagation_rate: float = 1,  # Local reproduction for next season
#     dispersal_efficiency: float = 0.5  # Spatial dispersal for next season
# ) -> dict:
#     """
#     Enhanced hatching simulation with daily spatial dispersal.
    
#     Key Changes in v3:
#     ------------------
#     - Daily dispersal occurs every day during the season
#     - dispersal_rate controls the fraction of survivors that disperse daily
#     - Remaining fraction (1 - dispersal_rate) stays in origin node
#     - Season-to-season propagation remains unchanged from v2
    
#     Parameters:
#     -----------
#     dd_df : pd.DataFrame
#         Degree-day data with DatetimeIndex and node columns
#     eggs_first_year : int, pd.Series, or dict
#         Initial egg counts for first year only
#     adjacency_matrix : np.ndarray
#         Square matrix (nodes x nodes) defining spatial connectivity.
#         Row-normalized internally. Matrix order must match dd_df columns.
#     dispersal_rate : float, default=0.1
#         Daily dispersal rate [0,1]. Fraction of survivors that disperse each day.
#         0 = no daily dispersal, 1 = all survivors disperse according to adjacency_matrix
#     propagation_rate : float, default=0.5
#         Local reproduction rate for season-to-season propagation
#     dispersal_efficiency : float, default=0.5
#         Spatial dispersal rate for season-to-season propagation
        
#     Returns:
#     --------
#     dict with keys: 'hatched', 'survivors', 'eggs_per_year', 'hatched_cum',
#     and optionally 'hazard', 'cumdd'
#     """
    
#     # Validate dispersal_rate
#     if not 0 <= dispersal_rate <= 1:
#         raise ValueError("dispersal_rate must be in [0, 1]")
    
#     # Validate adjacency matrix
#     if not isinstance(adjacency_matrix, np.ndarray):
#         raise ValueError("adjacency_matrix must be a numpy ndarray")
    
#     nodes = list(dd_df.columns)
#     n_nodes = len(nodes)
    
#     if adjacency_matrix.shape != (n_nodes, n_nodes):
#         raise ValueError(
#             f"adjacency_matrix must be square with shape ({n_nodes}, {n_nodes}) "
#             f"to match dd_df columns"
#         )
    
#     adj_matrix = adjacency_matrix.copy()
    
#     if np.any(adj_matrix < 0):
#         raise ValueError("Adjacency matrix cannot contain negative values")
    
#     # Row-normalize adjacency matrix
#     row_sums = np.sum(adj_matrix, axis=1, keepdims=True)
#     adj_matrix = np.divide(
#         adj_matrix, row_sums, 
#         out=np.zeros_like(adj_matrix), 
#         where=row_sums != 0
#     )
    
#     # Normalize season-to-season propagation parameters
#     param_sum = propagation_rate + dispersal_efficiency
#     if param_sum <= 0:
#         raise ValueError("propagation_rate + dispersal_efficiency must be > 0")
    
#     local_reproduction_rate = propagation_rate / param_sum
#     spatial_dispersal_rate = dispersal_efficiency / param_sum
    
#     dd_df = dd_df.sort_index()
#     X = dd_df.copy()
#     if clip_nonpositive:
#         X = X.clip(lower=0)

#     mode_norm = mode.lower()
#     if mode_norm not in {"cum", "daily", "mixed", "mixted"}:
#         raise ValueError("mode must be 'cum', 'daily', or 'mixed'/'mixted'.")

#     cumdd = None
#     if mode_norm in {"cum", "mixed", "mixted"}:
#         cumdd = X.groupby(X.index.year).cumsum()

#     def _to_series(p, name):
#         if p is None:
#             raise ValueError(f"Parameter '{name}' unexpectedly None after resolution.")
#         if np.isscalar(p):
#             return pd.Series({n: float(p) for n in nodes})
#         if isinstance(p, (pd.Series, dict)):
#             s = pd.Series(p, dtype=float)
#             missing = set(nodes) - set(s.index)
#             if missing:
#                 raise ValueError(
#                     f"Parameter '{name}' missing nodes: {sorted(missing)}"
#                 )
#             return s.loc[nodes].astype(float)
#         raise ValueError(
#             f"Parameter '{name}' must be scalar or Series/dict keyed by node."
#         )

#     def _resolve(override, base, name):
#         return _to_series(override if override is not None else base, name)

#     dd50_d = _resolve(dd50_daily, dd50, "dd50_daily")
#     k_d = _resolve(k_daily, k, "k_daily")
#     pmax_d = _resolve(p_max_daily, p_max, "p_max_daily")

#     dd50_c = _resolve(dd50_cum, dd50, "dd50_cum")
#     k_c = _resolve(k_cum, k, "k_cum")
#     pmax_c = _resolve(p_max_cum, p_max, "p_max_cum")

#     def _sigmoid(df, dd50_s, k_s):
#         Z = (df - dd50_s.values) * k_s.values
#         return 1.0 / (1.0 + np.exp(-Z))

#     # Calculate hazard
#     if mode_norm == "daily":
#         s_d = _sigmoid(X, dd50_d, k_d)
#         hazard = s_d.multiply(pmax_d.values, axis=1)
#     elif mode_norm == "cum":
#         s_c = _sigmoid(cumdd, dd50_c, k_c)
#         hazard = s_c.multiply(pmax_c.values, axis=1)
#     else:  # mixed
#         if mix_rule not in {"mean", "noisy_or"}:
#             raise ValueError("mix_rule must be 'mean' or 'noisy_or'.")
#         if mix_rule == "mean" and not (0.0 <= mix_alpha <= 1.0):
#             raise ValueError("mix_alpha must be in [0,1].")

#         s_d = _sigmoid(X, dd50_d, k_d)
#         s_c = _sigmoid(cumdd, dd50_c, k_c)
#         p_d = s_d.multiply(pmax_d.values, axis=1)
#         p_c = s_c.multiply(pmax_c.values, axis=1)

#         if mix_rule == "mean":
#             hazard = p_d * mix_alpha + p_c * (1.0 - mix_alpha)
#         else:  # noisy_or
#             hazard = 1.0 - (1.0 - p_d) * (1.0 - p_c)

#         hazard = hazard.clip(lower=0.0, upper=1.0)

#     if lag_days > 0:
#         mask = pd.Series(
#             dd_df.index.dayofyear <= lag_days, 
#             index=dd_df.index
#         )
#         hazard.loc[mask, :] = 0.0

#     def _process_eggs_input(eggs_input):
#         if isinstance(eggs_input, (int, np.integer, float)):
#             return pd.Series({n: int(eggs_input) for n in nodes})
#         if isinstance(eggs_input, pd.Series):
#             missing = set(nodes) - set(eggs_input.index)
#             if missing:
#                 raise ValueError(
#                     f"eggs_first_year Series missing nodes: {sorted(missing)}"
#                 )
#             return eggs_input.loc[nodes].astype(int)
#         if isinstance(eggs_input, dict):
#             s = pd.Series(eggs_input, dtype=int)
#             missing = set(nodes) - set(s.index)
#             if missing:
#                 raise ValueError(
#                     f"eggs_first_year dict missing nodes: {sorted(missing)}"
#                 )
#             return s.loc[nodes].astype(int)
#         raise ValueError(
#             "eggs_first_year must be int, Series, or dict keyed by node."
#         )

#     def _calculate_next_year_eggs(total_hatched_prev_year):
#         """Calculate eggs for next season using normalized propagation rates."""
#         local_eggs = (total_hatched_prev_year * local_reproduction_rate).astype(int)
        
#         dispersed_eggs = np.zeros(len(nodes))
#         hatched_array = total_hatched_prev_year.values * spatial_dispersal_rate
        
#         for i in range(len(nodes)):
#             dispersed_eggs[i] = np.sum(hatched_array * adj_matrix[i, :])
        
#         total_eggs = local_eggs.values + dispersed_eggs.astype(int)
#         return pd.Series(total_eggs, index=nodes, dtype=int)

#     def _apply_daily_dispersal(survivors):
#         """
#         Apply daily dispersal to current survivors.
        
#         dispersal_rate fraction disperses according to adjacency_matrix.
#         (1 - dispersal_rate) fraction remains in origin node.
#         """
#         # Fraction that disperses
#         dispersing = survivors * dispersal_rate
#         # Fraction that stays
#         staying = survivors * (1 - dispersal_rate)

#         # add gaussian noise to dispersing individuals
#         # noise = np.random.normal(0, 0.1, size=dispersing.shape) * dispersing * 0.5
#         # dispersing = np.clip(dispersing + noise, 0, None)
        
#         # Redistribute dispersing individuals
#         redistributed = np.zeros(len(survivors))
#         for i in range(len(survivors)):
#             redistributed[i] = np.sum(dispersing * adj_matrix[i, :])
        
#         # Total = staying in place + incoming from dispersal
#         total_survivors = staying + redistributed
#         return total_survivors.astype(int)

#     years = dd_df.index.year.unique()
#     rng = np.random.default_rng(random_state)
    
#     hatched_out, still_not_hatched_out = [], []
#     eggs_record = {}
    
#     current_eggs = _process_eggs_input(eggs_first_year)
#     eggs_record[years[0]] = current_eggs.copy()
    
#     for year_idx, y in enumerate(years):
#         idx_y = dd_df.index[dd_df.index.year == y]
#         H = pd.DataFrame(0, index=idx_y, columns=nodes, dtype=int)
#         S = pd.DataFrame(0, index=idx_y, columns=nodes, dtype=int)

#         still_not_hatched = current_eggs.astype(int).values
        
#         for t, day in enumerate(idx_y):
#             p = np.clip(hazard.loc[day].values, 0.0, 1.0)
#             hatch_t = rng.binomial(still_not_hatched, p)
#             still_not_hatched = still_not_hatched - hatch_t
            
#             # add gaussian noise to hatched individuals
#             noise = np.random.normal(0, 0.1, size=hatch_t.shape) * hatch_t * 0.5
#             hatch_t = np.clip(hatch_t + noise, 0, None).astype(int)

#             # Apply daily dispersal to unhatchd eggs
#             if dispersal_rate > 0:
#                 still_not_hatched = _apply_daily_dispersal(still_not_hatched)
            
#             H.iloc[t, :] = hatch_t
#             S.iloc[t, :] = still_not_hatched

#         hatched_out.append(H)
#         still_not_hatched_out.append(S)
        
#         # Calculate eggs for next year
#         if year_idx < len(years) - 1:
#             hatched_end = current_eggs - still_not_hatched
#             # current_eggs = _calculate_next_year_eggs(hatched_end)
#             # set new y eggs to the same as previous year hatched
#             current_eggs = current_eggs
#             eggs_record[years[year_idx + 1]] = current_eggs.copy()

#     hatched_df = pd.concat(hatched_out, axis=0)
#     survivors_df = pd.concat(still_not_hatched_out, axis=0)

#     out = {
#         "hatched": hatched_df,
#         "survivors": survivors_df,
#         "eggs_per_year": pd.DataFrame.from_dict(eggs_record, orient='index'),
#         "hatched_cum": hatched_df.groupby(hatched_df.index.year).cumsum()
#     }
    
#     if return_probs:
#         out["hazard"] = hazard
#     if cumdd is not None:
#         out["cumdd"] = cumdd
    
#     return out



import numpy as np
import pandas as pd

def sigmoid_curve(x, k=10, x0=0.5, a=500):
    # Returns sigmoid-shaped curve between 0 and a
    return a / (1 + np.exp(-k*(x-x0)))

def propagate_event(degree_df, adjacency, threshold, maxi=500, annual_reset=True, curve_k=10):
    """
    degree_df: DataFrame (index: dates, columns: nodes), values: degree days
    adjacency: NxN numpy array or DataFrame (adjacency matrix)
    threshold: float, degree day value for event appearance
    annual_reset: bool, resets event data every year
    curve_k: steepness parameter for sigmoid
    Returns: DataFrame, same shape as degree_df, with event occurrence values
    """
    nodes = degree_df.columns
    dates = degree_df.index
    event_df = pd.DataFrame(0, index=dates, columns=nodes)
    active = pd.DataFrame(False, index=dates, columns=nodes)
    last_year = None
    
    for t, date in enumerate(degree_df.index):
        year = pd.Timestamp(date).year
        # annual reset
        if annual_reset and (last_year is not None) and (year != last_year):
            active.iloc[t,:] = False
        last_year = year

        # Initiate event appearance if threshold is met
        for idx, node in enumerate(nodes):
            if not active.iloc[t, idx] and degree_df.iloc[t, idx] >= threshold:
                active.iloc[t, idx] = True

        # Propagate event through adjacency
        if t > 0:
            active.iloc[t,:] = active.iloc[t,:] | (
                (adjacency @ active.iloc[t-1,:].values.astype(int)) > 0
            )

        # Apply sigmoid curve, mapping degree_df (normalized) to occurrence
        norm_deg = np.clip((degree_df.iloc[t] - threshold) / (degree_df.max().max() - threshold), 0, 1)
        event_df.iloc[t,:] = sigmoid_curve(norm_deg, k=curve_k, a=maxi) * active.iloc[t,:].astype(float)

    return event_df


def generate_soil_quality(nodes_coordinates, positions_gaussians, spreads_gaussians, weights_gaussians):
    """
    this function generates a soil quality map based on gaussian distributions, then sample it at the nodes coordinates, which are given as a list of lat/lon tuples.
    gaussians mean are given in positions_gaussians, also in lat and lon,
    gaussians spreads are given in spreads_gaussians (in meters),
    gaussians max values are given in weights_gaussians
    the value taken is the max value of all gaussians at that point.
    """
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Compute distance in meters between two lat/lon points."""
        R = 6371000  # Earth radius in meters
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c

    nodes_coordinates = nodes_coordinates.values
    positions_gaussians = np.array(positions_gaussians)
    spreads_gaussians = np.array(spreads_gaussians)
    weights_gaussians = np.array(weights_gaussians)
    
    n_nodes = len(nodes_coordinates)
    n_gaussians = len(positions_gaussians)
    
    soil_quality = np.zeros(n_nodes)
    
    for i in range(n_nodes):
        node_lat, node_lon = nodes_coordinates[i]
        max_value = 0.0
        
        for j in range(n_gaussians):
            gauss_lat, gauss_lon = positions_gaussians[j]
            
            # Compute geodesic distance in meters
            distance = haversine_distance(node_lat, node_lon, gauss_lat, gauss_lon)
            distance_km = distance / 1000.0

            # Compute gaussian value
            variance = spreads_gaussians[j] ** 2
            gaussian_value = weights_gaussians[j] * np.exp(-distance_km**2 / (2 * variance))
            
            max_value = max(max_value, gaussian_value)
        
        soil_quality[i] = max_value
    
    return soil_quality