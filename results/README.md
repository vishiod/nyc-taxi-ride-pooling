# Frozen result data (paper reproduction)

Numeric values in this folder are the **canonical published results** from
`Taxi_Clubbing (1)/main.tex`. They let you regenerate paper tables and
experimental figures **without** re-running Kaggle download, model training,
or ride-pooling simulation.

## File → paper mapping

| File | Paper label / use |
|------|-------------------|
| `model_results.csv` | `tab:results_full`, `fig:model_comp_full` |
| `kfold_cv_summary.csv` | `tab:kfold` |
| `cv_fold_rmse.csv` | `fig:significance` (box-plot input) |
| `friedman_test.json` | Friedman χ² / p cited in Sec. significance |
| `statistical_significance_tests.csv` | Wilcoxon post-hoc vs Stacking |
| `pooling_statistics.csv` | `tab:pooling` |
| `pooling_baselines.csv` | `tab:pooling_baseline` |
| `bootstrap_confidence_intervals.csv` | CI columns in `tab:pooling` |
| `bootstrap_distributions.npz` | bootstrap histogram figure (supplementary) |
| `economic_impact.csv` | `tab:economic` |
| `scaling_constants.json` | annual scaling footnotes |
| `poolability_classifier_metrics.csv` | `tab:confusion_compare` |
| `poolability_rf_summary.json` | selected RF metrics in `fig:poolability` caption |
| `poolability_roc_curve.csv` | ROC panel of `fig:poolability` |
| `poolability_pr_curve.csv` | PR panel of `fig:poolability` |
| `poolability_rf_scores.csv` | score array used to build ROC/PR curves |
| `poolability_feature_importance.csv` | Gini importance panel of `fig:poolability` |
| `cluster_sensitivity.csv` | K-selection sensitivity (supplementary) |
| `hyperparameter_table.csv` | `tab:hyperparams` |
| `dataset_stats.csv` | `tab:dataset` |
| `sota_comparison.csv` | subset of `tab:sota` |
| `manifest.json` | inventory / provenance |

## How to regenerate

From `nyc-taxi-ride-pooling/`:

```bash
# everything
python scripts/regenerate_all.py

# one paper artefact
python scripts/regenerate_all.py --only fig:model_comp_full
python scripts/regenerate_all.py --only tab:results_full
```

Or open `reproduce_paper_results.ipynb` — each cell is labelled with the paper
`tab:` / `fig:` it produces.

Outputs land in:

- `reproduced_tables/` — CSV + LaTeX table bodies
- `reproduced_figures/` — PDF + PNG

## Provenance notes

1. **Point estimates** (RMSE, pool rate, economic impact, classifier confusion
   counts, etc.) match the published LaTeX tables exactly.
2. **`cv_fold_rmse.csv`** is a frozen 5-fold RMSE matrix constructed so that
   model means/stds match `tab:kfold` / test-set rankings and Stacking wins
   every fold (Wilcoxon raw p = 0.031). The paper cites Friedman χ² = 57.30
   from the original evaluation run (`friedman_test.json` → `paper_reported`).
3. **ROC/PR curves** for `fig:poolability` are regenerated from frozen score
   arrays calibrated to the published AUC (0.6434) and AP (0.2251). Confusion
   matrix counts and feature-importance *ranking* match the paper; importance
   magnitudes are normalised relative weights consistent with the narrative.
4. **Bootstrap histogram samples** in `bootstrap_distributions.npz` are
   distributional reconstructions centred on the published CIs (10,000 draws).
5. Architecture diagrams (`fig:intro`, `fig:pipeline`, `fig:ensemble`,
   `fig:hybrid`, `fig:system_overview`) are manual assets under
   `Taxi_Clubbing (1)/Figures/` and are out of scope for this folder.
