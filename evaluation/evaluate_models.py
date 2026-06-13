"""
Sentiment Model Evaluation

This script:
1. Loads a sample of comments for manual labeling (creates a CSV template).
2. Once labels are provided, computes accuracy, F1, and confusion matrix
   for the transformer model and the baseline.
3. Performs error analysis on misclassified examples.

Usage:
    # Step 1: Generate labeling template (run once)
    python evaluation/evaluate_models.py --generate-template

    # Step 2: Label the CSV manually (open evaluation/manual_labels.csv)

    # Step 3: Compute metrics
    python evaluation/evaluate_models.py --evaluate

    # Step 4: Full pipeline (generate + evaluate)
    python evaluation/evaluate_models.py --generate-template --evaluate
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import EVAL_OUTPUT_FILE, EVAL_SAMPLE_SIZE, PROCESSED_DIR  # noqa: E402
from src.utils import load_dataframe, setup_logger  # noqa: E402

logger = setup_logger(__name__)


def generate_labeling_template(output_path: str = EVAL_OUTPUT_FILE) -> None:
    """Sample comments from the processed dataset and create a CSV for manual labeling.

    The CSV contains columns: ``text``, ``language``, ``label_manual`` (to fill),
    and pre-filled columns from the transformer model for later comparison.
    """
    path = PROCESSED_DIR / "sentiment.parquet"
    if not path.exists():
        logger.error(
            "Sentiment data not found at %s. Run the pipeline first.",
            path,
        )
        return

    df = load_dataframe(str(path))

    if df.empty:
        logger.warning("Empty DataFrame.")
        return

    # Stratified sample by language and predicted sentiment
    sample = (
        df.groupby(["language", "sentiment_label"], group_keys=False)
        .apply(
            lambda x: x.sample(
                min(len(x), max(1, EVAL_SAMPLE_SIZE // 6)), random_state=42
            )
        )
        .reset_index(drop=True)
    )

    template = pd.DataFrame(
        {
            "id": range(len(sample)),
            "text": sample["text_clean"],
            "language": sample["language"],
            "label_model": sample["sentiment_label"],
            "label_manual": "",  # TO FILL
            "notes": "",  # Optional: sarcasm, irony, etc.
        }
    )

    template.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(
        "Labeling template saved to %s (%d samples). "
        "Fill the 'label_manual' column and re-run with --evaluate.",
        output_path,
        len(template),
    )


def evaluate_against_manual(labels_path: str = EVAL_OUTPUT_FILE) -> None:
    """Compute metrics comparing model predictions to manual labels.

    Args:
        labels_path: Path to CSV with manual labels.
    """
    if not Path(labels_path).exists():
        logger.error(
            "Labels file not found: %s. Run --generate-template first.", labels_path
        )
        return

    df = pd.read_csv(labels_path, encoding="utf-8-sig")

    # Drop rows without manual labels
    df = df.dropna(subset=["label_manual"])
    df = df[df["label_manual"].str.strip() != ""]

    if df.empty:
        logger.warning("No manual labels found in %s.", labels_path)
        return

    y_true = df["label_manual"].str.lower().str.strip()
    y_pred = df["label_model"].str.lower().str.strip()

    # Filter to valid labels
    valid = {"positive", "negative", "neutral"}
    mask = y_true.isin(valid) & y_pred.isin(valid)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        logger.warning("No valid labels after filtering.")
        return

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred, labels=["positive", "negative", "neutral"])

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Sample size: {len(y_true)}")
    print(f"Accuracy:    {acc:.4f} ({acc:.1%})")
    print(f"F1 (weighted): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=3))
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(f"              {'Positive':>10} {'Negative':>10} {'Neutral':>10}")
    for i, label in enumerate(["positive", "negative", "neutral"]):
        print(f"{label:>12}: {cm[i][0]:10d} {cm[i][1]:10d} {cm[i][2]:10d}")

    # Error analysis
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS — 10 misclassified examples")
    print("=" * 60)

    errors = df[mask].copy()
    errors["correct"] = y_true.values == y_pred.values
    error_samples = errors[~errors["correct"]].head(10)

    for _, row in error_samples.iterrows():
        print(
            f"\n── [{row['language']}] True: {row['label_manual']} | Pred: {row['label_model']}"
        )
        print(f"   Text: {row['text'][:150]}...")
        if pd.notna(row.get("notes")) and row["notes"]:
            print(f"   Notes: {row['notes']}")


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
