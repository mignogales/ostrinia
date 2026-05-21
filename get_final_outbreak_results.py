"""Outbreak-detection error aggregation across DL models and Degree-Day variants.

For every (dataset, model, variant, strategy) tuple this script loads the
per-seed predictions stored under ``paper_results/.../{seed}/predictions.npz``,
runs the configured outbreak-day-detection strategy, and records the mean
absolute day-error against the ground-truth outbreak day.

Path conventions assumed (edit `dl_pred_path` / `dd_pred_path` if yours differ):

    DL:  paper_results/{model}_{dataset}_nodes_embd_{embedding}/{seed}/predictions.npz
    DD:  paper_results/degree_day_{mapping}_pooled{pooled}_{dataset}/{seed}/predictions.npz

Note on the ``dataset`` argument to ``filter_nodes``: the original snippet
passes the loop variable (a string) into ``filter_nodes``. That is preserved
here verbatim. If your ``filter_nodes`` expects a loaded dataset object
instead, replace the call with the loader you already use elsewhere.
"""
import json
import os
from itertools import product
from typing import List, Optional

import numpy as np

from get_day_outbreak import (
    filter_nodes,
    get_days_outbreaks,
    compute_errors,
    get_curves,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DL_MODELS = ['gru', 'grugcn', 'mlp', 'transformer']
DD_MAPPINGS = ['linear', 'logistic', 'isotonic']
DD_POOLED = [True, False]

DATASETS = ['peakweather', 'ostrinia']
EMBEDDINGS = [False, True]                 # DL only
SEEDS = [42, 43, 44, 45, 46]

STRATEGIES = [
    'max_diff',
    'first_of_two_maxes',
    'max_after_day_th',
    'third_of_three_maxes',
    'first_of_two_maxes_after_day_150',
    'first_over_threshold',
    'dd25',
]

OUT_JSON = 'final_outbreak_results.json'


# ---------------------------------------------------------------------------
# Path builders -- edit if your run-dir naming differs
# ---------------------------------------------------------------------------
def dl_pred_path(model: str, dataset: str, embedding: bool, seed: int) -> str:
    return f'paper_results/{model}_{dataset}_nodes_embd_{embedding}/{seed}/predictions.npz'


def dd_pred_path(mapping: str, pooled: bool, dataset: str, seed: int) -> str:
    return f'paper_results/degree_day_{mapping}_pooled{pooled}_{dataset}/{seed}/predictions.npz'


# ---------------------------------------------------------------------------
# Core evaluation -- decoupled from path-building so DL and DD share it
# ---------------------------------------------------------------------------
def evaluate_outbreak_error(
    npz_files: List[str],
    dataset_name: str,
    strategy: str,
    th: int = 10,
) -> float:
    """Run the outbreak-day strategy on the ensemble of per-seed predictions.

    Returns the mean absolute day-error, or ``np.nan`` if no prediction files
    are available.
    """
    existing = [p for p in npz_files if os.path.exists(p)]
    if not existing:
        return float('nan')

    preds = get_curves([existing])
    # filter_nodes receives the dataset *name* here, matching the original
    # snippet's semantics. If your filter_nodes needs a loaded dataset, swap
    # the argument here.
    outbreak_nodes, day_outbreaks = filter_nodes(dataset_name, preds, 20)

    # Mean over seeds (axis 0) of the predicted curves
    y_pred_node_mean = np.mean(preds['y_pred_all'], axis=0)

    if strategy == 'max_after_day_th':
        th = 70 if dataset_name == 'ostrinia' else 150

    day_outbreaks_pred = get_days_outbreaks(
        y_pred_node_mean[None, :],
        outbreak_nodes,
        [0],
        strategy=strategy,
        threshold=th,
    )

    _, average_error, _ = compute_errors(
        outbreak_nodes, day_outbreaks, day_outbreaks_pred, [0]
    )
    return float(average_error)


def best_over_threshold_sweep(
    npz_files: List[str],
    dataset_name: str,
    max_th: int,
) -> float:
    """For the 'first_over_threshold' strategy, sweep th in [1, max_th] and
    return the minimum (best-case) error. Mirrors the original behaviour."""
    errs = []
    for th in range(1, max_th + 1):
        e = evaluate_outbreak_error(npz_files, dataset_name, 'first_over_threshold', th=th)
        if not (e != e):   # filter NaN
            errs.append(e)
    return min(errs) if errs else float('nan')


def evaluate_all_strategies(
    npz_files: List[str],
    dataset_name: str,
) -> dict:
    """Run every configured strategy against the same prediction ensemble."""
    out = {}
    for strategy in STRATEGIES:
        if strategy == 'first_over_threshold':
            max_th = 5 if dataset_name == 'ostrinia' else 25
            out[strategy] = best_over_threshold_sweep(npz_files, dataset_name, max_th)
        else:
            out[strategy] = evaluate_outbreak_error(npz_files, dataset_name, strategy)
    return out


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------
results: dict = {}

for dataset in DATASETS:
    results[dataset] = {}

    # ----- DL models -------------------------------------------------------
    for model in DL_MODELS:
        results[dataset][model] = {}
        for embedding in EMBEDDINGS:
            npz_files = [dl_pred_path(model, dataset, embedding, s) for s in SEEDS]
            results[dataset][model][str(embedding)] = evaluate_all_strategies(
                npz_files, dataset
            )

    # ----- Degree-Day variants --------------------------------------------
    # Each (mapping, pooled) is its own top-level model entry with a single
    # 'default' variant key, mirroring the shape used by DL models so that
    # downstream renderers stay uniform.
    for mapping, pooled in product(DD_MAPPINGS, DD_POOLED):
        label = f'degree_day_{mapping}_pooled{pooled}'
        results[dataset][label] = {}
        npz_files = [dd_pred_path(mapping, pooled, dataset, s) for s in SEEDS]
        results[dataset][label]['default'] = evaluate_all_strategies(
            npz_files, dataset
        )


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------
def _json_safe(o):
    # NaN -> None for portability across JSON readers
    if isinstance(o, float) and (o != o):
        return None
    # check also nan like this, not a nan np instance
    if isinstance(o, np.floating) and np.isnan(o):
        return None
    raise TypeError(f"Unserialisable object {o!r} of type {type(o)}")


with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2, default=_json_safe,
              allow_nan=True)

print(f"Saved outbreak results to {OUT_JSON}")