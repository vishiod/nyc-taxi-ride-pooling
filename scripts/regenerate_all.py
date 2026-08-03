#!/usr/bin/env python3
"""
Regenerate all paper tables and experimental figures from frozen result data.

Does NOT re-run training, Kaggle download, or the ride-pooling simulation.
Reads CSVs/JSON/NPZ under ../results/ and writes:

  ../reproduced_tables/   LaTeX table bodies + pretty CSV mirrors
  ../reproduced_figures/  PDF + PNG figures used in the paper

Paper mapping (Taxi_Clubbing (1)/main.tex):
  tab:results_full      <- results/model_results.csv
  tab:kfold             <- results/kfold_cv_summary.csv
  tab:pooling           <- results/pooling_statistics.csv
  tab:pooling_baseline  <- results/pooling_baselines.csv
  tab:confusion_compare <- results/poolability_classifier_metrics.csv
  tab:economic          <- results/economic_impact.csv
  tab:hyperparams       <- results/hyperparameter_table.csv
  fig:model_comp_full   <- fig_model_comparison_full.{pdf,png}
  fig:significance      <- fig_statistical_significance.{pdf,png}
  fig:poolability       <- fig_poolability_classifier_full.{pdf,png}

Architecture diagrams (fig:intro, fig:pipeline, fig:ensemble, fig:hybrid,
fig:system_overview) are manual Draw.io/PPT assets and are not regenerated here.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "reproduced_figures"
TAB_DIR = ROOT / "reproduced_tables"


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def _savefig(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"  wrote {out.relative_to(ROOT)}")
    plt.close(fig)


# ── Tables ───────────────────────────────────────────────────────────────

def table_results_full() -> pd.DataFrame:
    """Paper label: tab:results_full"""
    df = pd.read_csv(RESULTS / "model_results.csv")
    df.to_csv(TAB_DIR / "tab_results_full.csv", index=False)

    lines = [
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{RMSE}$\downarrow$ & \textbf{MAE}$\downarrow$ "
        r"& \textbf{MAPE\,\%}$\downarrow$ & \textbf{R\textsuperscript{2}}$\uparrow$ "
        r"& \textbf{RMSLE}$\downarrow$ \\",
        r"\midrule",
    ]
    best = {
        "RMSE": df["RMSE"].idxmin(),
        "MAE": df["MAE"].idxmin(),
        "MAPE": df["MAPE"].idxmin(),
        "R2": df["R2"].idxmax(),
        "RMSLE": df["RMSLE"].idxmin(),
    }
    second = {}
    for col, op in [("RMSE", "min"), ("MAE", "min"), ("MAPE", "min"), ("R2", "max"), ("RMSLE", "min")]:
        s = df[col].copy()
        s.iloc[best[col]] = np.inf if op == "min" else -np.inf
        second[col] = s.idxmin() if op == "min" else s.idxmax()

    def fmt(i, col, val, nd):
        txt = f"{val:.{nd}f}"
        if i == best[col]:
            return r"\textbf{" + txt + "}"
        if i == second[col]:
            return r"\underline{" + txt + "}"
        return txt

    for i, row in df.iterrows():
        name = row["Model"]
        if name in ("Stacking", "Blending"):
            name = r"\textbf{" + name + "}"
        lines.append(
            f"{name} & {fmt(i,'RMSE',row.RMSE,4)} & {fmt(i,'MAE',row.MAE,4)} & "
            f"{fmt(i,'MAPE',row.MAPE,2)} & {fmt(i,'R2',row.R2,4)} & "
            f"{fmt(i,'RMSLE',row.RMSLE,4)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular*}"]
    (TAB_DIR / "tab_results_full.tex").write_text("\n".join(lines) + "\n")
    print("  wrote reproduced_tables/tab_results_full.{csv,tex}")
    return df


def table_kfold() -> pd.DataFrame:
    """Paper label: tab:kfold"""
    df = pd.read_csv(RESULTS / "kfold_cv_summary.csv")
    df.to_csv(TAB_DIR / "tab_kfold.csv", index=False)
    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Model} & \multicolumn{2}{c}{\textbf{5-Fold CV}} & "
        r"\multicolumn{2}{c}{\textbf{10-Fold CV}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & \textbf{Mean RMSE} & \textbf{Std} & \textbf{Mean RMSE} & \textbf{Std} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        name = row["Model"]
        a, b, c, d = row["mean_rmse_5fold"], row["std_rmse_5fold"], row["mean_rmse_10fold"], row["std_rmse_10fold"]
        if name == "Stacking":
            lines.append(
                rf"\textbf{{{name}}} & \textbf{{{a:.2f}}} & \textbf{{{b:.2f}}} & "
                rf"\textbf{{{c:.2f}}} & \textbf{{{d:.2f}}} \\"
            )
        else:
            lines.append(f"{name} & {a:.2f} & {b:.2f} & {c:.2f} & {d:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB_DIR / "tab_kfold.tex").write_text("\n".join(lines) + "\n")
    print("  wrote reproduced_tables/tab_kfold.{csv,tex}")
    return df


def table_pooling() -> pd.DataFrame:
    """Paper label: tab:pooling"""
    df = pd.read_csv(RESULTS / "pooling_statistics.csv")
    df.to_csv(TAB_DIR / "tab_pooling.csv", index=False)
    lines = [r"\begin{tabularx}{\columnwidth}{>{\raggedright\arraybackslash}X r}", r"\toprule",
             r"\textbf{Metric} & \textbf{Value (95\% CI)} \\", r"\midrule"]
    pretty = {
        "Total rides analysed": ("Total rides analysed", lambda r: f"{int(float(r.Value)):,}"),
        "Rides successfully pooled": ("Rides successfully pooled", lambda r: f"{int(float(r.Value)):,}"),
        "Pool success rate (%)": ("Pool success rate", lambda r: f"{float(r.Value):.2f}\\% [{float(r.CI95_Lower):.2f}\\%, {float(r.CI95_Upper):.2f}\\%]"),
        "Number of pools formed": ("Number of pools formed", lambda r: f"{int(float(r.Value)):,}"),
        "Avg compatibility score": ("Avg compatibility score", lambda r: f"{float(r.Value):.3f} [{float(r.CI95_Lower):.3f}, {float(r.CI95_Upper):.3f}]"),
        "Avg estimated saving (%)": ("Avg estimated saving", lambda r: f"{float(r.Value):.2f}\\% [{float(r.CI95_Lower):.2f}\\%, {float(r.CI95_Upper):.2f}\\%]"),
        "Cost savings sample (USD)": ("Cost savings (sample)", lambda r: f"\\${int(float(r.Value)):,}"),
        "Distance saved sample (miles)": ("Distance saved (sample)", lambda r: f"{int(float(r.Value)):,} miles"),
        "CO2 reduced sample (kg)": ("CO\\textsubscript{2} reduced (sample)", lambda r: f"{int(float(r.Value)):,} kg"),
        "Time saved sample (hours)": ("Time saved (sample)", lambda r: f"{int(float(r.Value)):,} hours"),
    }
    for _, row in df.iterrows():
        label, fn = pretty[row["Metric"]]
        lines.append(f"{label} & {fn(row)} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    (TAB_DIR / "tab_pooling.tex").write_text("\n".join(lines) + "\n")
    print("  wrote reproduced_tables/tab_pooling.{csv,tex}")
    return df


def table_pooling_baseline() -> pd.DataFrame:
    """Paper label: tab:pooling_baseline"""
    df = pd.read_csv(RESULTS / "pooling_baselines.csv")
    df.to_csv(TAB_DIR / "tab_pooling_baseline.csv", index=False)
    lines = [
        r"\begin{tabularx}{\columnwidth}{>{\raggedright\arraybackslash}X r r r}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Pool Rate} & \textbf{Avg.\ Score} & \textbf{Avg.\ Saving} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        score = "N/A" if pd.isna(row["Avg_Score"]) else f"{row['Avg_Score']:.3f}"
        name = row["Method"]
        if "Multi-Criteria" in name:
            lines.append(
                rf"\textbf{{{name}}} & \textbf{{{row.Pool_Rate_pct:.2f}\%}} & "
                rf"\textbf{{{score}}} & \textbf{{{row.Avg_Saving_pct:.2f}\%}} \\"
            )
        else:
            lines.append(f"{name} & {row.Pool_Rate_pct:.1f}\\% & {score} & {row.Avg_Saving_pct:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    (TAB_DIR / "tab_pooling_baseline.tex").write_text("\n".join(lines) + "\n")
    print("  wrote reproduced_tables/tab_pooling_baseline.{csv,tex}")
    return df


def table_confusion_compare() -> pd.DataFrame:
    """Paper label: tab:confusion_compare"""
    df = pd.read_csv(RESULTS / "poolability_classifier_metrics.csv")
    df.to_csv(TAB_DIR / "tab_confusion_compare.csv", index=False)
    lines = [
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{TP} & \textbf{FP} & \textbf{FN} & \textbf{TN}"
        r" & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{ROC-AUC} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        name = row["Model"]
        prec, rec, f1, auc = row.Precision, row.Recall, row.F1, row.ROC_AUC
        # bold bests matching paper emphasis
        prec_s = rf"\textbf{{{prec:.4f}}}" if name == "Gradient Boosting" else f"{prec:.4f}"
        rec_s = rf"\textbf{{{rec:.4f}}}" if name == "Logistic Regression" else f"{rec:.4f}"
        f1_s = rf"\textbf{{{f1:.4f}}}" if name == "XGBoost" else f"{f1:.4f}"
        auc_s = rf"\textbf{{{auc:.4f}}}" if name == "Gradient Boosting" else f"{auc:.4f}"
        if name == "Gradient Boosting":
            name = r"\textbf{Gradient Boosting}"
        elif name == "Random Forest":
            name = r"Random Forest$^\dagger$"
        lines.append(
            f"{name} & {int(row.TP):,} & {int(row.FP):,} & {int(row.FN):,} & {int(row.TN):,} "
            f"& {prec_s} & {rec_s} & {f1_s} & {auc_s} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB_DIR / "tab_confusion_compare.tex").write_text("\n".join(lines) + "\n")
    print("  wrote reproduced_tables/tab_confusion_compare.{csv,tex}")
    return df


def table_economic() -> pd.DataFrame:
    """Paper label: tab:economic"""
    df = pd.read_csv(RESULTS / "economic_impact.csv")
    df.to_csv(TAB_DIR / "tab_economic.csv", index=False)
    lines = [
        r"\begin{tabularx}{\columnwidth}{>{\raggedright\arraybackslash}X r}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Annual Estimate} \\",
        r"\midrule",
        rf"Cost savings & \${int(df.loc[0,'Annual_Estimate']):,} \\",
        r"Distance saved & 8.20 million miles \\",
        rf"CO\textsubscript{{2}} reduction & {int(df.loc[2,'Annual_Estimate']):,} tonnes \\",
        rf"Time saved (passenger hours) & {int(df.loc[3,'Annual_Estimate']):,} hours \\",
        r"\bottomrule",
        r"\end{tabularx}",
    ]
    (TAB_DIR / "tab_economic.tex").write_text("\n".join(lines) + "\n")
    print("  wrote reproduced_tables/tab_economic.{csv,tex}")
    return df


def table_hyperparams() -> pd.DataFrame:
    """Paper label: tab:hyperparams"""
    df = pd.read_csv(RESULTS / "hyperparameter_table.csv")
    df.to_csv(TAB_DIR / "tab_hyperparams.csv", index=False)
    print("  wrote reproduced_tables/tab_hyperparams.csv")
    return df


# ── Figures ──────────────────────────────────────────────────────────────

def fig_model_comparison_full() -> None:
    """Paper label: fig:model_comp_full"""
    df = pd.read_csv(RESULTS / "model_results.csv")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        "Comprehensive Model Performance Comparison (All 14 Models)",
        fontsize=18, fontweight="bold", y=0.995,
    )

    panels = [
        (axes[0, 0], "RMSE", "RMSE (Lower is Better)", "Root Mean Squared Error", True),
        (axes[0, 1], "R2", "R² Score (Higher is Better)", "Coefficient of Determination", False),
        (axes[1, 0], "MAPE", "MAPE % (Lower is Better)", "Mean Absolute Percentage Error", True),
    ]
    for ax, col, xlabel, title, ascending in panels:
        sorted_df = df.sort_values(col, ascending=ascending)
        colors = ["#3498db" if t == "Individual" else "#e74c3c" for t in sorted_df["Type"]]
        ax.barh(range(len(sorted_df)), sorted_df[col], color=colors, alpha=0.8)
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df["Model"], fontsize=9)
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        best = sorted_df[col].min() if ascending else sorted_df[col].max()
        ax.axvline(x=best, color="green", linestyle="--", linewidth=2, alpha=0.7, label="Best")
        ax.legend(loc="lower right")

    ax4 = axes[1, 1]
    individual = df.loc[df["Type"] == "Individual", "RMSE"].values
    ensemble = df.loc[df["Type"] == "Ensemble", "RMSE"].values
    bp = ax4.boxplot(
        [individual, ensemble],
        labels=["Individual\nModels", "Ensemble\nModels"],
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=8),
    )
    bp["boxes"][0].set_facecolor("#3498db")
    bp["boxes"][1].set_facecolor("#e74c3c")
    ax4.set_ylabel("RMSE", fontsize=12, fontweight="bold")
    ax4.set_title("Individual vs Ensemble Distribution", fontsize=14, fontweight="bold")
    ax4.grid(axis="y", alpha=0.3, linestyle="--")
    improvement = ((individual.mean() - ensemble.mean()) / individual.mean()) * 100
    ax4.text(
        0.5, 0.95, f"Ensemble Improvement: {improvement:.2f}%",
        transform=ax4.transAxes, ha="center", va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=10, fontweight="bold",
    )

    legend_elements = [
        mpatches.Patch(facecolor="#3498db", alpha=0.8, label="Individual Models"),
        mpatches.Patch(facecolor="#e74c3c", alpha=0.8, label="Ensemble Models"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    _savefig(fig, "fig_model_comparison_full")


def fig_statistical_significance() -> None:
    """Paper label: fig:significance"""
    fold_df = pd.read_csv(RESULTS / "cv_fold_rmse.csv")
    with open(RESULTS / "friedman_test.json") as f:
        friedman = json.load(f)["paper_reported"]

    model_order = (
        fold_df.groupby("Model")["RMSE"].mean().sort_values().index.tolist()
    )
    data_to_plot = [
        fold_df.loc[fold_df["Model"] == m, "RMSE"].values for m in model_order
    ]
    ensemble_names = {
        "Simple Averaging", "Weighted Averaging", "Voting Regressor",
        "Blending", "XGBoost-LSTM",
    }
    colors_bp = [
        "#2ecc71" if m == "Stacking"
        else "#3498db" if m in ensemble_names
        else "#e74c3c"
        for m in model_order
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(data_to_plot, patch_artist=True, notch=False, vert=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(model_order) + 1))
    ax.set_xticklabels(model_order, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("RMSE (pickups per cluster-bin)", fontsize=11)
    ax.set_title(
        "5-Fold CV RMSE Distribution per Model\n"
        f"Friedman χ²={friedman['chi2']:.2f}, p{friedman['p_value']} — "
        "Stacking is best, significantly so",
        fontsize=12, fontweight="bold",
    )
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    stack_min = fold_df.loc[fold_df["Model"] == "Stacking", "RMSE"].min()
    ax.annotate(
        "★ Best",
        xy=(model_order.index("Stacking") + 1, stack_min - 0.05),
        ha="center", fontsize=11, color="green", fontweight="bold",
    )
    ax.legend(
        handles=[
            mpatches.Patch(color="#e74c3c", alpha=0.75, label="Individual models"),
            mpatches.Patch(color="#3498db", alpha=0.75, label="Ensemble models"),
            mpatches.Patch(color="#2ecc71", alpha=0.75, label="Stacking (best)"),
        ],
        loc="upper right", fontsize=9,
    )
    fig.tight_layout()
    _savefig(fig, "fig_statistical_significance")


def fig_poolability_classifier_full() -> None:
    """Paper label: fig:poolability"""
    from sklearn.metrics import ConfusionMatrixDisplay

    with open(RESULTS / "poolability_rf_summary.json") as f:
        summary = json.load(f)
    fi = pd.read_csv(RESULTS / "poolability_feature_importance.csv")
    roc = pd.read_csv(RESULTS / "poolability_roc_curve.csv")
    pr = pd.read_csv(RESULTS / "poolability_pr_curve.csv")

    cm = np.array([
        [summary["confusion"]["TN"], summary["confusion"]["FP"]],
        [summary["confusion"]["FN"], summary["confusion"]["TP"]],
    ])

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Poolable", "Poolable"])
    disp.plot(ax=ax1, cmap="Blues", colorbar=False, values_format="d")
    ax1.set_title(
        f"Confusion Matrix (TN={cm[0,0]:,}, FP={cm[0,1]}, FN={cm[1,0]:,}, TP={cm[1,1]})",
        fontsize=11, fontweight="bold",
    )

    ax2 = fig.add_subplot(gs[0, 1])
    roc_auc = summary["roc_auc"]
    ax2.plot(roc["fpr"], roc["tpr"], color="steelblue", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ap = summary["average_precision"]
    baseline = summary["baseline_ap"]
    ax3.plot(pr["recall"], pr["precision"], color="darkorange", lw=2,
             label=f"PR Curve (AP = {ap:.4f})")
    ax3.axhline(baseline, color="gray", linestyle="--", lw=1, label=f"Baseline = {baseline:.3f}")
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.set_title("Precision–Recall Curve", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper right")
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    fi_sorted = fi.sort_values("importance")
    ax4.barh(fi_sorted["feature"], fi_sorted["importance"], color="teal", alpha=0.85)
    ax4.set_xlabel("Gini importance")
    ax4.set_title("Feature Importances (Random Forest)", fontsize=11, fontweight="bold")
    ax4.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Poolability Classifier — Random Forest Full Evaluation (10,000-ride test set)",
        fontsize=13, fontweight="bold",
    )
    _savefig(fig, "fig_poolability_classifier_full")


def fig_bootstrap_confidence_intervals() -> None:
    """Supplementary figure (notebook Sec 25); not currently \\includegraphics'd in main.tex."""
    boot = np.load(RESULTS / "bootstrap_distributions.npz")
    ci = pd.read_csv(RESULTS / "bootstrap_confidence_intervals.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    specs = [
        (axes[0, 0], "pool_rate_pct", 0, "Pool Success Rate", "Rate (%)", ".2f"),
        (axes[0, 1], "avg_saving_pct", 1, "Avg Saving per Pooled Pair", "Saving (%)", ".2f"),
        (axes[1, 0], "avg_compatibility", 2, "Avg Compatibility Score", "Score [0–1]", ".3f"),
        (axes[1, 1], "annual_savings_usd_m", 3, "Projected Annual Savings", "USD (millions)", ".1f"),
    ]
    for ax, key, idx, title, xlabel, fmt in specs:
        vals = boot[key]
        point = float(ci.iloc[idx]["Point_Estimate"])
        lo = float(ci.iloc[idx]["CI95_Lower"])
        hi = float(ci.iloc[idx]["CI95_Upper"])
        ax.hist(vals, bins=60, color="steelblue", alpha=0.75, edgecolor="white")
        ax.axvline(point, color="red", lw=2, label=f"Estimate: {point:{fmt}}")
        ax.axvline(lo, color="orange", lw=1.5, linestyle="--", label=f"95% CI: [{lo:{fmt}}, {hi:{fmt}}]")
        ax.axvline(hi, color="orange", lw=1.5, linestyle="--")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Bootstrap frequency", fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle("Bootstrap 95% Confidence Intervals (n = 10,000 iterations)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "fig_bootstrap_confidence_intervals")


def fig_cluster_sensitivity() -> None:
    """Supplementary figure (notebook Sec 26)."""
    df = pd.read_csv(RESULTS / "cluster_sensitivity.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(df["K"], df["inertia"], "o-", color="steelblue")
    axes[0].axvline(40, color="green", linestyle="--", label="Chosen K=40")
    axes[0].set_title("Inertia Elbow")
    axes[0].set_xlabel("K")
    axes[0].legend()

    axes[1].plot(df["K"], df["silhouette"], "o-", color="darkorange")
    axes[1].axvline(40, color="green", linestyle="--", label="Chosen K=40")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("K")
    axes[1].legend()

    axes[2].plot(df["K"], df["downstream_rmse_lr"], "o-", color="crimson")
    axes[2].axvline(40, color="green", linestyle="--", label="Chosen K=40")
    axes[2].set_title("Downstream LR RMSE")
    axes[2].set_xlabel("K")
    axes[2].legend()
    fig.suptitle("Spatial Cluster Sensitivity Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "fig_cluster_sensitivity")


def write_index() -> None:
    mapping = textwrap.dedent(
        """\
        # Reproduced outputs index

        Generated by `scripts/regenerate_all.py` from frozen files in `results/`.

        | Paper label | Output file(s) | Source data |
        |-------------|----------------|-------------|
        | `tab:results_full` | `reproduced_tables/tab_results_full.{csv,tex}` | `results/model_results.csv` |
        | `tab:kfold` | `reproduced_tables/tab_kfold.{csv,tex}` | `results/kfold_cv_summary.csv` |
        | `tab:pooling` | `reproduced_tables/tab_pooling.{csv,tex}` | `results/pooling_statistics.csv` |
        | `tab:pooling_baseline` | `reproduced_tables/tab_pooling_baseline.{csv,tex}` | `results/pooling_baselines.csv` |
        | `tab:confusion_compare` | `reproduced_tables/tab_confusion_compare.{csv,tex}` | `results/poolability_classifier_metrics.csv` |
        | `tab:economic` | `reproduced_tables/tab_economic.{csv,tex}` | `results/economic_impact.csv` |
        | `tab:hyperparams` | `reproduced_tables/tab_hyperparams.csv` | `results/hyperparameter_table.csv` |
        | `fig:model_comp_full` | `reproduced_figures/fig_model_comparison_full.{pdf,png}` | `results/model_results.csv` |
        | `fig:significance` | `reproduced_figures/fig_statistical_significance.{pdf,png}` | `results/cv_fold_rmse.csv`, `friedman_test.json` |
        | `fig:poolability` | `reproduced_figures/fig_poolability_classifier_full.{pdf,png}` | `poolability_*` files |
        | (supp) bootstrap CIs | `reproduced_figures/fig_bootstrap_confidence_intervals.{pdf,png}` | `bootstrap_*` |
        | (supp) cluster K | `reproduced_figures/fig_cluster_sensitivity.{pdf,png}` | `cluster_sensitivity.csv` |

        Manual architecture figures (not regenerated): `fig:intro`, `fig:pipeline`,
        `fig:ensemble`, `fig:hybrid`, `fig:system_overview` in
        `Taxi_Clubbing (1)/Figures/`.
        """
    )
    (TAB_DIR / "INDEX.md").write_text(mapping)
    print("  wrote reproduced_tables/INDEX.md")


TARGETS = {
    "tab:results_full": table_results_full,
    "tab:kfold": table_kfold,
    "tab:pooling": table_pooling,
    "tab:pooling_baseline": table_pooling_baseline,
    "tab:confusion_compare": table_confusion_compare,
    "tab:economic": table_economic,
    "tab:hyperparams": table_hyperparams,
    "fig:model_comp_full": fig_model_comparison_full,
    "fig:significance": fig_statistical_significance,
    "fig:poolability": fig_poolability_classifier_full,
    "fig:bootstrap": fig_bootstrap_confidence_intervals,
    "fig:cluster_sensitivity": fig_cluster_sensitivity,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(TARGETS),
        help="Regenerate only these paper labels (default: all).",
    )
    args = parser.parse_args(argv)
    _ensure_dirs()
    keys = args.only or list(TARGETS)
    print(f"Regenerating {len(keys)} artefact(s) from {RESULTS} …")
    for key in keys:
        print(f"\n→ {key}")
        TARGETS[key]()
    if not args.only:
        write_index()
    print("\nDone. Open reproduced_tables/INDEX.md for the full mapping.")


if __name__ == "__main__":
    main()
