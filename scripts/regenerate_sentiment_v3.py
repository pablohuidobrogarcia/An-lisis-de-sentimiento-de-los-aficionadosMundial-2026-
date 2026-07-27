"""
Regenerate sentiment predictions with the fine-tuned model
(models/sentiment_finetuned_v1/). Reads the unified v3 dataset,
processes all rows, and saves to v3_final.

Usage:
    python scripts/regenerate_sentiment_v3.py
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils import setup_logger

LOG_FILE = _project_root / "logs" / "regenerate_sentiment_v3.log"
logger = setup_logger("regenerate_sentiment_v3", log_file=LOG_FILE)

INPUT_PARQUET = (
    _project_root
    / "data"
    / "processed"
    / "comentarios_sentimiento"
    / "comentarios_sentimiento_v3.parquet"
)
CHECKPOINT_DIR = INPUT_PARQUET.parent / ".checkpoints_v3"
PROGRESS_FILE = CHECKPOINT_DIR / "progress.json"
OUTPUT_NEW = INPUT_PARQUET.with_name("comentarios_sentimiento_v3_final.parquet")

MODEL_PATH = str(_project_root / "models" / "sentiment_finetuned_v1")
MODEL_VERSION = "cardiffnlp-finetuned-v1|epoch4|DDA4BD3225D1"
BATCH_SIZE = 5000
MAX_LEN = 128
ID2LABEL = {0: "NEG", 1: "NEU", 2: "POS"}
LABEL2ID = {"NEG": 0, "NEU": 1, "POS": 2}
DEVICE = torch.device("cpu")

# ── Fine-tuned model (lazy-loaded singleton) ──────────────────────────────

_TOKENIZER = None
_MODEL = None


def _ensure_model():
    global _TOKENIZER, _MODEL
    if _MODEL is not None:
        return _TOKENIZER, _MODEL
    logger.info("Loading fine-tuned model from %s ...", MODEL_PATH)
    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
    _MODEL = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    _MODEL.eval()
    logger.info("Fine-tuned model loaded. Label mapping: %s", ID2LABEL)
    return _TOKENIZER, _MODEL


# ── Batch inference ────────────────────────────────────────────────────────


INFERENCE_BATCH = 64  # inner batch size to avoid OOM on CPU


def _process_batch(texts, batch_size=INFERENCE_BATCH):
    tokenizer, model = _ensure_model()
    all_preds = []
    all_probas = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)

        for j in range(len(batch_texts)):
            p = probs[j]
            probas = {
                "negative": round(float(p[0]), 6),
                "neutral": round(float(p[1]), 6),
                "positive": round(float(p[2]), 6),
            }
            label = max(probas, key=probas.get).upper()[:3]
            all_preds.append(label)
            all_probas.append(str(probas))

    return all_preds, all_probas


def _process_baseline(
    df: pd.DataFrame, text_col: str = "text_clean", lang_col: str = "language"
):
    from src.sentiment import predict_sentiment_baseline

    labels = []
    for i in range(len(df)):
        lang = df.iloc[i][lang_col] if lang_col in df.columns else "en"
        try:
            s = predict_sentiment_baseline(df.iloc[i][text_col], lang)
            label = max(s, key=s.get).upper()[:3]
        except Exception:
            label = "NEU"
        labels.append(label)
    return labels


def regenerate(df_full: pd.DataFrame, is_test: bool = False):
    """Process rows in batches with checkpoint/resume."""
    total = len(df_full)

    progress = load_progress()
    if progress["completed_chunks"] and not is_test:
        done_ids = set()
        for ci in progress["completed_chunks"]:
            cp = _chunk_path(ci)
            if cp.exists():
                done_ids.update(pd.read_parquet(cp)["comment_id"].tolist())
        mask = ~df_full["comment_id"].isin(done_ids)
        df_pending = df_full[mask].reset_index(drop=True)
        skipped = total - len(df_pending)
        if skipped:
            logger.info(
                "Resume: %d rows already done, %d pending", skipped, len(df_pending)
            )
        start_chunk = max(progress["completed_chunks"]) + 1
    else:
        df_pending = df_full
        progress = {
            "completed_chunks": [],
            "total_chunks": (total + BATCH_SIZE - 1) // BATCH_SIZE,
            "started_at": str(pd.Timestamp.now()),
            "model": MODEL_VERSION,
            "is_test": is_test,
        }
        save_progress(progress)
        start_chunk = 0

    n_pending = len(df_pending)
    if n_pending == 0:
        logger.info("Nothing to process.")
        return

    n_batches = (n_pending + BATCH_SIZE - 1) // BATCH_SIZE
    t_start = time.time()

    for batch_idx in range(start_chunk, start_chunk + n_batches):
        lo = (batch_idx - start_chunk) * BATCH_SIZE
        hi = min(lo + BATCH_SIZE, n_pending)
        batch = df_pending.iloc[lo:hi].copy()

        t_b0 = time.time()
        try:
            texts = batch["text_clean"].tolist()
            bert_labels, bert_probas = _process_batch(texts)
            base_labels = _process_baseline(batch)
        except Exception as exc:
            logger.error("Chunk %d failed: %s", batch_idx, exc)
            raise

        result = batch[["comment_id"]].copy()
        result["sentiment_bert"] = bert_labels
        result["sentiment_bert_probas"] = bert_probas
        result["sentiment_baseline"] = base_labels
        result["sentiment_model_version"] = MODEL_VERSION

        _atomic_write(result, _chunk_path(batch_idx))

        progress["completed_chunks"].append(batch_idx)
        save_progress(progress)

        rows_done = (
            sum(1 for ci in range(batch_idx + 1) if _chunk_path(ci).exists())
            * BATCH_SIZE
        )
        if rows_done > total:
            rows_done = total

        elapsed = time.time() - t_b0
        rate = elapsed / len(batch)
        eta = rate * (n_pending - hi)

        logger.info(
            "Chunk %d/%d | %d rows | %.1fs | %.2fs/row | %d/%d done | ETA %s",
            batch_idx + 1,
            progress["total_chunks"],
            len(batch),
            elapsed,
            rate,
            rows_done,
            total,
            f"{eta/60:.1f}min" if eta < 3600 else f"{eta/3600:.1f}h",
        )

    logger.info(
        "All done in %.1fs (%.2f min)",
        time.time() - t_start,
        (time.time() - t_start) / 60,
    )


# ── Checkpoint helpers ─────────────────────────────────────────────────────


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    part_path = path.with_suffix(".parquet.part")
    df.to_parquet(part_path, index=False)
    os.replace(str(part_path), str(path))


def _chunk_path(chunk_idx: int) -> Path:
    return CHECKPOINT_DIR / f"chunk_{chunk_idx:05d}.parquet"


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"completed_chunks": [], "total_chunks": 0}


def save_progress(progress: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, default=str)


def merge_chunks(up_to_chunk: int) -> pd.DataFrame:
    chunks = []
    for i in range(up_to_chunk):
        cp = _chunk_path(i)
        if cp.exists():
            chunks.append(pd.read_parquet(cp))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# ── Verification ────────────────────────────────────────────────────────────


def verify_result(df_final: pd.DataFrame) -> dict:
    issues = {}
    n_dups = df_final["comment_id"].duplicated().sum()
    if n_dups:
        issues["duplicated_comment_ids"] = int(n_dups)
    n_nan_version = df_final["sentiment_model_version"].isna().sum()
    if n_nan_version:
        issues["nan_sentiment_model_version"] = int(n_nan_version)
    n_nan_bert = df_final["sentiment_bert"].isna().sum()
    if n_nan_bert:
        issues["nan_sentiment_bert"] = int(n_nan_bert)
    versions = df_final["sentiment_model_version"].value_counts().to_dict()
    return {
        "total_rows": len(df_final),
        "unique_comment_ids": int(df_final["comment_id"].nunique()),
        "expected_version": MODEL_VERSION,
        "version_distribution": {str(k): int(v) for k, v in versions.items()},
        "issues": issues,
    }


def show_verification(df_final: pd.DataFrame, n: int = 20):
    sample = df_final.sample(n=min(n, len(df_final)), random_state=99)
    print(f"\n{'=' * 100}")
    print(f"  VERIFICATION: {len(sample)} random rows")
    print(f"{'=' * 100}")
    for _, row in sample.iterrows():
        text = row.get("text_clean", "")
        if len(str(text)) > 120:
            text = str(text)[:120] + "..."
        print(
            f"\n  sentiment={row['sentiment_bert']}  version={row['sentiment_model_version']}"
        )
        print(
            f"       text:  {str(text).encode('ascii', errors='replace').decode('ascii')}"
        )
    print()


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate sentiment v3 with fine-tuned model"
    )
    parser.add_argument("--test", action="store_true", help="Test on a small sample")
    parser.add_argument("--n", type=int, default=2000, help="Rows for test mode")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Replace original parquet with this result",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing checkpoints and start fresh",
    )
    args = parser.parse_args()

    if not INPUT_PARQUET.exists():
        logger.error("Input not found: %s", INPUT_PARQUET)
        sys.exit(1)

    if args.apply and not args.test:
        if not OUTPUT_NEW.exists():
            logger.error("v3_final not found: %s", OUTPUT_NEW)
            sys.exit(1)
        shutil.copy2(
            OUTPUT_NEW, INPUT_PARQUET.with_name("comentarios_sentimiento.parquet")
        )
        logger.info("Overwritten: comentarios_sentimiento.parquet")
        print("Original parquet replaced with v3_final.")
        return

    if args.reset:
        if CHECKPOINT_DIR.exists():
            shutil.rmtree(CHECKPOINT_DIR)
            logger.info("Checkpoints deleted (fresh start).")

    df_full = pd.read_parquet(INPUT_PARQUET)
    logger.info("Base data: %d rows, %d columns", len(df_full), len(df_full.columns))

    if args.test:
        n = min(args.n, len(df_full))
        df_full = df_full.sample(n=n, random_state=42).reset_index(drop=True)
        logger.info("Test mode: sampling %d rows", n)

    regenerate(df_full, is_test=args.test)

    progress = load_progress()
    n_chunks_done = len(progress.get("completed_chunks", []))
    if n_chunks_done == 0:
        logger.warning("No chunks were processed.")
        return

    df_result = merge_chunks(max(progress["completed_chunks"]) + 1)
    logger.info("Result merged: %d rows", len(df_result))

    if args.test:
        print(f"\n{'=' * 60}")
        print(f"  TEST COMPLETE: {len(df_result)} rows processed")
        print(f"  Model: {MODEL_VERSION}")
        print(f"{'=' * 60}")
        return

    # Assemble final parquet — drop old sentiment columns from base first
    merge_cols = [
        "comment_id",
        "sentiment_bert",
        "sentiment_bert_probas",
        "sentiment_baseline",
        "sentiment_model_version",
    ]
    old_sentiment_cols = [
        c for c in merge_cols if c != "comment_id" and c in df_full.columns
    ]
    if old_sentiment_cols:
        df_full = df_full.drop(columns=old_sentiment_cols)
    df_result = df_result[merge_cols].drop_duplicates(subset="comment_id")
    df_final = df_full.merge(df_result, on="comment_id", how="left")
    _atomic_write(df_final, OUTPUT_NEW)

    logger.info(
        "Saved to %s (%d rows, %d cols)",
        OUTPUT_NEW,
        len(df_final),
        len(df_final.columns),
    )

    # Verification
    ver = verify_result(df_final)
    print(f"\n{'=' * 60}")
    print("  REGENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Input:          {INPUT_PARQUET.name}")
    print(f"  Output:         {OUTPUT_NEW.name}")
    print(f"  Model:          {MODEL_VERSION}")
    print(f"  Rows:           {ver['total_rows']:,}")
    print(f"  Unique IDs:     {ver['unique_comment_ids']:,}")
    print(f"  Version dist:   {ver['version_distribution']}")
    if ver["issues"]:
        print(f"  ISSUES:         {ver['issues']}")
    else:
        print("  ISSUES:         NONE (all OK)")
    print(f"  Original:       {INPUT_PARQUET.name} untouched")
    print("  Use --apply to replace original comentarios_sentimiento.parquet")
    print(f"{'=' * 60}")

    show_verification(df_final)

    # ── Evaluation against test set ──────────────────────────
    print(f"{'=' * 60}")
    print("  EVALUATION vs TEST SET (500 held-out)")
    print(f"{'=' * 60}")
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
    )

    DATA_EVAL = _project_root / "data" / "eval"
    test_raw = pd.read_excel(DATA_EVAL / "sentiment_labeling_test.xlsx")
    test_raw["manual_label"] = (
        test_raw["manual_label"].astype(str).str.strip().str.upper()
    )
    test_merged = test_raw.merge(
        df_final[["comment_id", "sentiment_bert"]], on="comment_id", how="inner"
    )
    if len(test_merged) == 500:
        y_true = test_merged["manual_label"].map(LABEL2ID).values
        y_pred = test_merged["sentiment_bert"].map(LABEL2ID).values
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        kappa = cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2])
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        f1_per = f1_score(y_true, y_pred, average=None)
        print(f"  Accuracy:       {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Macro-F1:       {f1_macro:.4f}")
        print(f"  Cohen's Kappa:  {kappa:.4f}")
        print(f"  F1 NEG:         {f1_per[0]:.4f}")
        print(f"  F1 NEU:         {f1_per[1]:.4f}")
        print(f"  F1 POS:         {f1_per[2]:.4f}")
        print("  Confusion Matrix:")
        print("           Pred NEG  Pred NEU  Pred POS")
        print(f"  Real NEG    {cm[0][0]:>5}     {cm[0][1]:>5}     {cm[0][2]:>5}")
        print(f"  Real NEU    {cm[1][0]:>5}     {cm[1][1]:>5}     {cm[1][2]:>5}")
        print(f"  Real POS    {cm[2][0]:>5}     {cm[2][1]:>5}     {cm[2][2]:>5}")
    else:
        print(f"  Warning: test set mismatch ({len(test_merged)}/500 found)")
    print()


if __name__ == "__main__":
    main()
