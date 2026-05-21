#!/usr/bin/env bash
set -euo pipefail

# Quick server runner for the paper pipeline.
#
# Stage 1: experiments/results_paper.py trains/evaluates a model and writes:
#   paper_results/<run_tag>/<seed>/test_results.json
#   paper_results/<run_tag>/<seed>/predictions.npz
#
# Stage 2: get_final_results.py and get_final_outbreak_results.py aggregate
# those saved files.
#
# Usage examples:
#   bash scripts/run_server_pipeline.sh
#   CONDA_ENV=taming-env bash scripts/run_server_pipeline.sh
#   DATASETS="ostrinia peakweather" bash scripts/run_server_pipeline.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-taming-env}"
DATASETS="${DATASETS:-ostrinia}"
PYTHON_CMD=(python)

if command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV" python)
fi

run_experiment() {
  echo
  echo "==> $*"
  "${PYTHON_CMD[@]}" experiments/results_paper.py "$@"
}

echo "Repository: $ROOT_DIR"
echo "Python: ${PYTHON_CMD[*]}"
echo "Datasets: $DATASETS"

# ---------------------------------------------------------------------------
# Degree-day baseline grid
# ---------------------------------------------------------------------------
# These names match config/model/degree_day.yaml and the aggregation scripts:
#   paper_results/degree_day_<mapping>_pooled<True|False>_<dataset>/...
for dataset in $DATASETS; do
  for mapping in linear logistic isotonic; do
    for pooled in true false; do
      run_experiment \
        "model=degree_day" \
        "dataset=${dataset}" \
        "model.hparams.mapping=${mapping}" \
        "model.hparams.pooled=${pooled}"
    done
  done
done

# ---------------------------------------------------------------------------
# Optional: deep-learning paper grid
# ---------------------------------------------------------------------------
# Uncomment this block if you want to regenerate the neural baselines too.
# It can be expensive because each call loops over seeds 42..51 internally.
#
# for dataset in $DATASETS; do
#   for model in gru grugcn mlp transformer; do
#     for use_node_embeddings in false true; do
#       run_experiment \
#         "model=${model}" \
#         "dataset=${dataset}" \
#         "model.hparams.use_node_embeddings=${use_node_embeddings}"
#     done
#   done
# done

# ---------------------------------------------------------------------------
# Aggregate saved results
# ---------------------------------------------------------------------------
echo
echo "==> Aggregating test metrics"
"${PYTHON_CMD[@]}" get_final_results.py

echo
echo "==> Aggregating outbreak-day errors"
"${PYTHON_CMD[@]}" get_final_outbreak_results.py

echo
echo "Done. Main outputs:"
echo "  paper_results/"
echo "  final_results_summary.json"
echo "  final_results_table.tex"
echo "  final_outbreak_results.json"
