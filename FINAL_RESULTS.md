# PICL

Physics-informed causal learning for transformer fault diagnosis from DGA.
Companion code for the JFI 2026 paper. Tested on Python 3.10+ / CPU.

## Install

    pip install -r requirements.txt

## Single run (seed 42)

    python train.py

Writes `results/results_summary.json` plus the saved model under
`results/models/`. Runs end-to-end in ~30 s on CPU.

## Five-seed run + downstream experiments

    PYTHONPATH=. python experiments/run_seeds.py --seeds 42 43 44 45 46
    PYTHONPATH=. python experiments/main_table.py
    PYTHONPATH=. python experiments/knowledge_alignment.py
    PYTHONPATH=. python experiments/coverage_risk_curves.py
    PYTHONPATH=. python experiments/reliability_diagram.py
    PYTHONPATH=. python experiments/confusion_matrix.py
    PYTHONPATH=. python experiments/counterfactual_faithfulness.py
    PYTHONPATH=. python experiments/ablation.py

All scripts write CSV tables to `tables/` and figures (PDF) to `figures/`.
Per-seed artifacts (model state, score arrays) live under
`results/seeds/seed_<S>/`.

The shipped `results/seeds/seed_*/results_summary.json` files and the
pre-built `tables/*.csv` / `figures/*.pdf` are the reference outputs
referenced by `FINAL_RESULTS.md`. Re-running the scripts above replaces
them with fresh ones.

## Layout

    train.py                    entry point for one seed
    config/                     YAML hyperparameters + IEC 60599 prior
    data/                       1648-sample DGA dataset
    picl/                       library (graph, scm, learn, augment,
                                inference, classifier head, trainer)
    experiments/                downstream evaluation scripts
    results/seeds/seed_<S>/     per-seed results_summary.json
    tables/                     CSV outputs
    figures/                    PDF figures
    FINAL_RESULTS.md            results write-up
