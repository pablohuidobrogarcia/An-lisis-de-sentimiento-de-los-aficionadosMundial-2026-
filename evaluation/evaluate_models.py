"""
Sentiment Model Evaluation

Computes accuracy, F1, confusion matrices, and error analysis for both
the BERT model and the baseline against manually labeled data.

Usage:
    # Step 1: Generate labeling template (run after sentiment pipeline)
    python evaluation/evaluate_models.py --generate-template

    # Step 2: Label the CSV manually (open evaluation/manual_labels_template.csv)

    # Step 3: Compute metrics
    python evaluation/evaluate_models.py --evaluate
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import EVAL_SAMPLE_SIZE  # noqa: E402
from src.utils import setup_logger  # noqa: E402

logger = setup_logger(__name__)

LABELS_PATH = Path(__file__).resolve().parent / "manual_labels_template.csv"
RESULTS_PATH = Path(__file__).resolve().parent / "EVALUATION_RESULTS.md"
CM_DIR = Path(__file__).resolve().parent


def generate_labeling_template(output_path: str = str(LABELS_PATH)) -> None:
    """Generate a CSV template for manual labeling."""
    from evaluation.create_labeling_sample import generate_labeling_sample

    generate_labeling_sample(output_path=output_path, n_samples=EVAL_SAMPLE_SIZE)


def _plot_confusion_matrix(
    cm,
    labels,
    title,
    filename,
    normalize: bool = False,
) -> None:
    """Plot and save a confusion matrix."""
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_display = cm
        fmt = "d"
        vmax = cm.max()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                format(cm_display[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm_display[i, j] > 0.5 * vmax else "black",
            )
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved to %s", filename)


def _extract_labels(df: pd.DataFrame) -> tuple:
    """Extract true and predicted labels, trying multiple column names."""
    # True labels
    if "manual_label" in df.columns:
        y_true = df["manual_label"].str.strip().str.upper()
    else:
        return None, None

    # Model predicted labels — try new names first
    for col in [
        "sentiment_bert_predicted",
        "sentiment_bert",
        "sentiment_label",
        "label_model",
    ]:
        if col in df.columns:
            y_pred_model = df[col].str.strip().str.upper()
            break
    else:
        return None, None

    # Baseline predicted labels
    y_pred_base = None
    for col in [
        "sentiment_baseline_predicted",
        "sentiment_baseline",
        "sentiment_baseline_label",
    ]:
        if col in df.columns:
            y_pred_base = df[col].str.strip().str.upper()
            break

    return (y_true, y_pred_model, y_pred_base)


def evaluate_against_manual(labels_path: str = str(LABELS_PATH)) -> None:
    """Compute metrics comparing model predictions to manual labels.

    Evaluates both BERT and baseline (if available), generates confusion
    matrix plots, and writes a markdown summary.
    """
    if not Path(labels_path).exists():
        logger.error("Labels file not found: %s.", labels_path)
        return

    df = pd.read_csv(labels_path, encoding="utf-8-sig")

    # Drop rows without manual labels
    df = df.dropna(subset=["manual_label"])
    df = df[df["manual_label"].str.strip() != ""]

    if df.empty:
        logger.warning("No manual labels found.")
        return

    extracted = _extract_labels(df)
    if extracted is None or extracted[0] is None:
        logger.error("Could not find required columns.")
        return

    y_true, y_pred_model, y_pred_base = extracted
    valid = {"POS", "NEG", "NEU"}
    labels_sorted = ["POS", "NEG", "NEU"]

    # Filter to valid labels
    mask = y_true.isin(valid) & y_pred_model.isin(valid)
    y_true_f = y_true[mask]
    y_pred_model_f = y_pred_model[mask]
    y_pred_base_f = y_pred_base[mask] if y_pred_base is not None else None

    if len(y_true_f) == 0:
        logger.warning("No valid labels after filtering.")
        return

    # ── Metrics ──────────────────────────────────────────────────────────
    def compute_metrics(true, pred, name):
        acc = accuracy_score(true, pred)
        f1_w = f1_score(true, pred, average="weighted")
        f1_m = f1_score(true, pred, average="macro")
        prec_m = precision_score(true, pred, average="macro")
        rec_m = recall_score(true, pred, average="macro")
        cm = confusion_matrix(true, pred, labels=labels_sorted)
        report = classification_report(true, pred, labels=labels_sorted, digits=3)
        return {
            "name": name,
            "accuracy": acc,
            "f1_weighted": f1_w,
            "f1_macro": f1_m,
            "precision_macro": prec_m,
            "recall_macro": rec_m,
            "confusion_matrix": cm,
            "report": report,
            "n": len(true),
        }

    model_metrics = compute_metrics(y_true_f, y_pred_model_f, "BERT (pysentimiento)")

    results_text = []
    results_text.append("# Evaluation Results\n")
    results_text.append(f"Sample size: **{model_metrics['n']}** comments\n")
    results_text.append("## BERT (pysentimiento)\n")
    results_text.append(
        f"- Accuracy: {model_metrics['accuracy']:.3f} ({model_metrics['accuracy']:.1%})"
    )
    results_text.append(f"- F1 (weighted): {model_metrics['f1_weighted']:.3f}")
    results_text.append(f"- F1 (macro): {model_metrics['f1_macro']:.3f}")
    results_text.append(f"- Precision (macro): {model_metrics['precision_macro']:.3f}")
    results_text.append(f"- Recall (macro): {model_metrics['recall_macro']:.3f}")
    results_text.append("")
    results_text.append("### Classification Report\n")
    results_text.append("```")
    results_text.append(model_metrics["report"])
    results_text.append("```\n")

    # Confusion matrix for BERT
    cm_path = CM_DIR / "confusion_matrix_bert.png"
    _plot_confusion_matrix(
        model_metrics["confusion_matrix"],
        labels_sorted,
        "BERT — Confusion Matrix",
        str(cm_path),
    )
    results_text.append("![BERT Confusion Matrix](confusion_matrix_bert.png)\n")

    # ── Print results ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BERT (pysentimiento) — EVALUATION")
    print("=" * 60)
    print(f"  Sample size: {model_metrics['n']}")
    print(
        f"  Accuracy:    {model_metrics['accuracy']:.3f} ({model_metrics['accuracy']:.1%})"
    )
    print(f"  F1 (weighted): {model_metrics['f1_weighted']:.3f}")
    print(f"  F1 (macro):    {model_metrics['f1_macro']:.3f}")
    print("\n  Classification Report:")
    print(model_metrics["report"])

    # ── Baseline metrics ─────────────────────────────────────────────────
    if y_pred_base_f is not None:
        base_mask = y_pred_base_f.isin(valid)
        base_metrics = compute_metrics(
            y_true_f[base_mask],
            y_pred_base_f[base_mask],
            "Baseline",
        )
        results_text.append("## Baseline\n")
        results_text.append(
            f"- Accuracy: {base_metrics['accuracy']:.3f} ({base_metrics['accuracy']:.1%})"
        )
        results_text.append(f"- F1 (weighted): {base_metrics['f1_weighted']:.3f}")
        results_text.append(f"- F1 (macro): {base_metrics['f1_macro']:.3f}")
        results_text.append("")
        results_text.append("### Classification Report\n")
        results_text.append("```")
        results_text.append(base_metrics["report"])
        results_text.append("```\n")

        cm_path_base = CM_DIR / "confusion_matrix_baseline.png"
        _plot_confusion_matrix(
            base_metrics["confusion_matrix"],
            labels_sorted,
            "Baseline — Confusion Matrix",
            str(cm_path_base),
        )
        results_text.append(
            "![Baseline Confusion Matrix](confusion_matrix_baseline.png)\n"
        )

        print("\n" + "=" * 60)
        print("  BASELINE — EVALUATION")
        print("=" * 60)
        print(f"  Sample size: {base_metrics['n']}")
        print(
            f"  Accuracy:    {base_metrics['accuracy']:.3f} ({base_metrics['accuracy']:.1%})"
        )
        print(f"  F1 (weighted): {base_metrics['f1_weighted']:.3f}")
        print("  Classification Report:")
        print(base_metrics["report"])

    # ── Error analysis ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ERROR ANALYSIS — 10 BERT misclassifications")
    print("=" * 60)

    df_err = df[mask].copy()
    df_err["correct"] = y_true_f.values == y_pred_model_f.values
    error_samples = df_err[~df_err["correct"]].head(10)

    results_text.append("## Error Analysis\n")
    results_text.append("Top 10 BERT misclassifications:\n")
    results_text.append("| # | Language | True | Predicted | Text |")
    results_text.append("|---|----------|------|-----------|------|")

    for i, (_, row) in enumerate(error_samples.iterrows(), 1):
        text_short = str(row.get("text_clean", row.get("text", "")))[:120]
        print(
            f"\n  [{i}] Lang: {row['language']} | True: {y_true_f.iloc[i-1]} | "
            f"Pred: {y_pred_model_f.iloc[i-1]}"
        )
        print(f"      Text: {text_short}")
        results_text.append(
            f"| {i} | {row['language']} | {y_true_f.iloc[i-1]} | "
            f"{y_pred_model_f.iloc[i-1]} | {text_short} |"
        )

    results_text.append("")

    # Write summary markdown
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(results_text))
    logger.info("Evaluation results saved to %s", RESULTS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentiment model evaluation")
    parser.add_argument(
        "--generate-template",
        action="store_true",
        help="Generate CSV template for manual labeling",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate model predictions against manual labels",
    )
    args = parser.parse_args()

    if not (args.generate_template or args.evaluate):
        parser.print_help()
        return

    if args.generate_template:
        generate_labeling_template()

    if args.evaluate:
        evaluate_against_manual()


if __name__ == "__main__":
    main()
