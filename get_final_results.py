"""Aggregate paper_results JSON outputs across models, datasets and seeds.

Computes mean +- std per metric for:
    * Deep-learning models, split by ``use_node_embeddings`` in {False, True}.
    * Degree-Day baseline, split by (mapping, pooled).

The two families share a single ``results`` dict with a generic "variant"
axis so that downstream JSON and LaTeX renderers stay uniform.

Run directory naming conventions assumed (edit ``dl_run_path`` /
``dd_run_path`` if yours differ):

    paper_results/{model}_{dataset}_nodes_embd_{use_embd}/{seed}/test_results.json
    paper_results/degree_day_{mapping}_pooled{pooled}_{dataset}/{seed}/test_results.json
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("paper_results")

DL_MODELS: List[str] = ["gru", "grugcn", "mlp", "transformer"]
DATASETS: List[str] = ["peakweather", "ostrinia"]
SEEDS: List[int] = [42, 43, 44, 45, 46]
METRICS: List[str] = ["test_mae", "test_mre", "test_mse"]

# Deep-learning configurations
DL_USE_NODE_EMBD: List[bool] = [False, True]

# Degree-Day configurations
DD_MAPPINGS: List[str] = ["linear", "logistic", "isotonic"]
DD_POOLED: List[bool] = [True, False]   # set to [True] only if no per-node runs

OUT_JSON = Path("final_results_summary.json")
OUT_TEX = Path("final_results_table.tex")


# ---------------------------------------------------------------------------
# Path builders -- edit these if your run-dir naming differs
# ---------------------------------------------------------------------------
def dl_run_path(model: str, dataset: str, use_embd: bool, seed: int) -> Path:
    return (
        RESULTS_DIR
        / f"{model}_{dataset}_nodes_embd_{use_embd}"
        / str(seed)
        / "test_results.json"
    )


def dd_run_path(dataset: str, mapping: str, pooled: bool, seed: int) -> Path:
    return (
        RESULTS_DIR
        / f"degree_day_{mapping}_pooled{pooled}_{dataset}"
        / str(seed)
        / "test_results.json"
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _aggregate(values_by_metric: Dict[str, List[float]]) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for metric, values in values_by_metric.items():
        if values:
            out[metric] = {
                "mean": float(np.mean(values)),
                "std":  float(np.std(values)),
                "n":    len(values),
            }
        else:
            out[metric] = {"mean": None, "std": None, "n": 0}
    return out


def _read_seeds(path_fn, seeds: List[int]) -> Dict[str, List[float]]:
    acc = {m: [] for m in METRICS}
    for seed in seeds:
        fp = path_fn(seed)
        if not fp.exists():
            print(f"[missing] {fp}")
            continue
        with open(fp) as f:
            data = json.load(f)
        for m in METRICS:
            if m in data:
                acc[m].append(float(data[m]))
            else:
                print(f"[warn]    metric '{m}' not in {fp}")
    return acc


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------
results: Dict[str, Dict[str, Dict[str, Dict]]] = {}

# Deep-learning models -------------------------------------------------------
for model in DL_MODELS:
    results[model] = {}
    for dataset in DATASETS:
        results[model][dataset] = {}
        for use_embd in DL_USE_NODE_EMBD:
            variant = f"nodes_embd_{use_embd}"
            acc = _read_seeds(
                lambda s, m=model, d=dataset, e=use_embd: dl_run_path(m, d, e, s),
                SEEDS,
            )
            results[model][dataset][variant] = _aggregate(acc)

# Degree-Day baseline --------------------------------------------------------
for mapping, pooled in product(DD_MAPPINGS, DD_POOLED):
    model_label = f"degree_day_{mapping}_{'pooled' if pooled else 'pernode'}"
    results[model_label] = {}
    for dataset in DATASETS:
        results[model_label][dataset] = {}
        # Single variant per (mapping, pooled): keep the nesting depth
        # consistent with DL models so the table renderer is uniform.
        variant = "default"
        acc = _read_seeds(
            lambda s, d=dataset, mp=mapping, pl=pooled: dd_run_path(d, mp, pl, s),
            SEEDS,
        )
        results[model_label][dataset][variant] = _aggregate(acc)

# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=4)
print(json.dumps(results, indent=4))


# ---------------------------------------------------------------------------
# LaTeX table (booktabs, publication-quality)
# ---------------------------------------------------------------------------
def fmt(d: Dict[str, Optional[float]]) -> str:
    if d["mean"] is None or d["std"] is None:
        return "N/A"
    # 4-decimal mean +- std; switch to scientific notation for tiny stds
    return f"{d['mean']:.4f} $\\pm$ {d['std']:.4f}"


def _pretty(name: str) -> str:
    return name.replace("_", r"\_")


cols = "l l l " + " ".join("c" for _ in METRICS)
lines: List[str] = [
    r"\begin{tabular}{" + cols + "}",
    r"\toprule",
    "Model & Dataset & Variant & " + " & ".join(_pretty(m) for m in METRICS) + r" \\",
    r"\midrule",
]
for model, by_dataset in results.items():
    for dataset, by_variant in by_dataset.items():
        for variant, metric_vals in by_variant.items():
            cells = " & ".join(fmt(metric_vals[m]) for m in METRICS)
            lines.append(
                f"{_pretty(model)} & {_pretty(dataset)} & {_pretty(variant)} & {cells} \\\\"
            )
    lines.append(r"\midrule")
# Replace the trailing \midrule with \bottomrule
lines[-1] = r"\bottomrule"
lines.append(r"\end{tabular}")

with open(OUT_TEX, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"LaTeX table saved to {OUT_TEX}")