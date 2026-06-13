"""
Generate a stratified sample of comments for manual sentiment labeling.

Usage:
    python evaluation/create_labeling_sample.py

Outputs ``evaluation/manual_labels_template.csv`` with columns:
    comment_id, text_clean, language, sentiment_bert_predicted, manual_label

The sample is stratified across predicted labels and language to ensure
coverage of all classes.
"""

import sys
from pathlib import Path

import pandas as pd

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import EVAL_SAMPLE_SIZE, PROCESSED_DIR  # noqa: E402
from src.utils import load_dataframe, setup_logger  # noqa: E402

logger = setup_logger(__name__)

LABELS_PATH = Path(__file__).resolve().parent / "manual_labels_template.csv"


def generate_labeling_sample(
    output_path: str = str(LABELS_PATH),
    n_samples: int = EVAL_SAMPLE_SIZE,
) -> None:
    """Create a CSV template for manual sentiment labeling.

    Samples are stratified by ``language`` and ``sentiment_bert`` predicted
    label so the evaluation set is not skewed toward the majority class.

    Args:
        output_path: Where to write the CSV.
        n_samples: Target number of samples (~120-150).
    """
    # Try new path first, then legacy
    candidates = [
        Path("data/processed/comentarios_sentimiento/comentarios_sentimiento.parquet"),
        Path("data/processed/comentarios_sentimiento.parquet"),
        PROCESSED_DIR / "sentiment.parquet",
        PROCESSED_DIR / "preprocessed.parquet",
        PROCESSED_DIR / "comentarios_limpios/comentarios_limpios.parquet",
    ]
    df = None
    for p in candidates:
        if p.exists():
            df = load_dataframe(str(p))
            logger.info("Loaded data from %s", p)
            break

    if df is None or df.empty:
        logger.error("No processed sentiment data found. Run the pipeline first.")
        return

    # Detect available columns
    lang_col = "language" if "language" in df.columns else None
    pred_col = (
        "sentiment_bert"
        if "sentiment_bert" in df.columns
        else "sentiment_label"
        if "sentiment_label" in df.columns
        else None
    )
    text_col = "text_clean" if "text_clean" in df.columns else "text"
    id_col = "comment_id" if "comment_id" in df.columns else None

    if not lang_col or not pred_col:
        logger.error("Required columns missing (language, sentiment). Aborting.")
        return

    # Stratified sampling
    strata_cols = [lang_col]
    if pred_col:
        strata_cols.append(pred_col)

    n_per_group = max(1, n_samples // (df.groupby(strata_cols).ngroups or 1))

    sample = (
        df.groupby(strata_cols, group_keys=False)
        .apply(lambda x: x.sample(min(len(x), n_per_group), random_state=42))
        .reset_index(drop=True)
    )

    # Cap at n_samples
    if len(sample) > n_samples:
        sample = sample.sample(n=n_samples, random_state=42).reset_index(drop=True)

    template = pd.DataFrame(
        {
            "comment_id": sample[id_col].values if id_col else range(len(sample)),
            "text_clean": sample[text_col].values,
            "language": sample[lang_col].values,
            "sentiment_bert_predicted": sample[pred_col].values if pred_col else "",
            "manual_label": "",  # TO FILL
        }
    )

    template.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(
        "Labeling template saved to %s (%d samples). "
        "Fill the 'manual_label' column with POS / NEG / NEU.",
        output_path,
        len(template),
    )


if __name__ == "__main__":
    generate_labeling_sample()
