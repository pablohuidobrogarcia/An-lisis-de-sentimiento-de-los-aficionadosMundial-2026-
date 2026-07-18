"""
Regenerate sentiment predictions with cardiffnlp/twitter-xlm-roberta-base-sentiment.

Usage:
    # Test run on 2000 rows (no overwrite, prints 20 verification cases)
    python scripts/regenerate_sentiment.py --test --n 2000

    # Full run on all rows (checkpointed, resumes on interrupt)
    python scripts/regenerate_sentiment.py

    # After review, replace original parquet
    python scripts/regenerate_sentiment.py --apply
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils import setup_logger

LOG_FILE = _project_root / "logs" / "regenerate_sentiment.log"
logger = setup_logger("regenerate_sentiment", log_file=LOG_FILE)

SENTIMENT_PARQUET = (
    _project_root
    / "data"
    / "processed"
    / "comentarios_sentimiento"
    / "comentarios_sentimiento.parquet"
)
CHECKPOINT_DIR = SENTIMENT_PARQUET.parent / ".checkpoints"
PROGRESS_FILE = CHECKPOINT_DIR / "progress.json"
OUTPUT_NEW = SENTIMENT_PARQUET.with_name("comentarios_sentimiento_v2.parquet")

SENTIMENT_COLS = [
    "sentiment_bert",
    "sentiment_bert_probas",
    "sentiment_baseline",
    "sentiment_model_version",
]
BATCH_SIZE = 5000

# ── Single cardiffnlp pipeline (lazy-loaded singleton) ──────────────────────

_PIPE = None
_MODEL_VERSION = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
_LABEL_MAP = {}


def _ensure_pipeline():
    global _PIPE, _LABEL_MAP
    if _PIPE is not None:
        return _PIPE
    from transformers import pipeline

    logger.info("Loading cardiffnlp/twitter-xlm-roberta-base-sentiment ...")
    _PIPE = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        max_length=512,
        truncation=True,
    )
    id2label = _PIPE.model.config.id2label
    _LABEL_MAP = {}
    for k, v in id2label.items():
        _LABEL_MAP[v.lower()] = v.upper()[:3]
    logger.info("CardiffNLP model loaded. Label mapping: %s", _LABEL_MAP)
    return _PIPE


# ── Batch inference function ─────────────────────────────────────────────────


def _process_with_cardiffnlp(
    df: pd.DataFrame, text_col: str = "text_clean", lang_col: str = "language"
) -> pd.DataFrame:
    """Run cardiffnlp/xlm-roberta on all texts (multilingual, single pass).

    Returns df with columns: sentiment_bert, sentiment_bert_probas,
    sentiment_baseline, sentiment_model_version.
    """
    pipe = _ensure_pipeline()
    texts = df[text_col].tolist()
    n = len(texts)

    results = pipe(texts, top_k=None)
    # results is list of list of dicts: [[{"label":"negative","score":...},...],...]

    bert_labels = []
    bert_probas = []

    for result_list in results:
        scores = {}
        for item in result_list:
            label_key = _LABEL_MAP.get(item["label"].lower(), "NEU")
            scores[label_key] = item["score"]
        # Ensure all 3 keys exist
        pos = scores.get("POS", 0.0)
        neg = scores.get("NEG", 0.0)
        neu = scores.get("NEU", 0.0)
        total = pos + neg + neu
        if total > 0:
            pos /= total
            neg /= total
            neu /= total
        probas = {
            "positive": round(pos, 6),
            "negative": round(neg, 6),
            "neutral": round(neu, 6),
        }
        label = max(probas, key=probas.get).upper()[:3]
        bert_labels.append(label)
        bert_probas.append(str(probas))

    # ── Baseline (VADER for EN, lexicon for ES) ──
    from src.sentiment import predict_sentiment_baseline

    base_labels = []
    for i in range(n):
        lang = df.iloc[i][lang_col] if lang_col in df.columns else "en"
        try:
            s = predict_sentiment_baseline(df.iloc[i][text_col], lang)
            label = max(s, key=s.get).upper()[:3]
        except Exception:
            label = "NEU"
        base_labels.append(label)

    result = df.copy()
    result["sentiment_bert"] = bert_labels
    result["sentiment_bert_probas"] = bert_probas
    result["sentiment_baseline"] = base_labels
    result["sentiment_model_version"] = _MODEL_VERSION

    return result


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to path with atomic .part → rename pattern."""
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


# ── Processing ────────────────────────────────────────────────────────────────


def regenerate(df_full: pd.DataFrame, is_test: bool = False, test_n: int = 0):
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
            "model": _MODEL_VERSION,
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
            result = _process_with_cardiffnlp(batch)
        except Exception as exc:
            logger.error("Chunk %d failed: %s", batch_idx, exc)
            raise

        elapsed = time.time() - t_b0
        _atomic_write(result, _chunk_path(batch_idx))

        progress["completed_chunks"].append(batch_idx)
        save_progress(progress)

        rows_done = (
            sum(1 for ci in range(batch_idx + 1) if _chunk_path(ci).exists())
            * BATCH_SIZE
        )
        if rows_done > total:
            rows_done = total

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


# ── Verification ──────────────────────────────────────────────────────────────


def show_verification(df_old: pd.DataFrame, df_new: pd.DataFrame, n: int = 20):
    merged = df_old.merge(
        df_new[["comment_id", "sentiment_bert", "sentiment_model_version"]],
        on="comment_id",
        suffixes=("_old", "_new"),
    )
    merged.columns = [
        "comment_id",
        "text_clean",
        "language",
        "sentiment_old",
        "sentiment_new",
        "model_version",
    ]
    merged = merged.dropna(subset=["sentiment_old", "sentiment_new"])

    sample = merged.sample(n=min(n, len(merged)), random_state=99)
    changes = (sample["sentiment_old"] != sample["sentiment_new"]).sum()

    print(f"\n{'='*100}")
    print(f"  VERIFICATION: {len(sample)} random cases  |  {changes} differ from old")
    print(f"{'='*100}")
    for _, row in sample.iterrows():
        text = row["text_clean"]
        if len(text) > 120:
            text = text[:120] + "..."
        match = "[OK]" if row["sentiment_old"] == row["sentiment_new"] else "[DIF]"
        text_ascii = text.encode("ascii", errors="replace").decode("ascii")
        print(
            f"\n  {match} lang={row['language']}  old={row['sentiment_old']}  new={row['sentiment_new']}"
        )
        print(f"       model: {row['model_version']}")
        print(f"       text:  {text_ascii}")


# ── Main ──────────────────────────────────────────────────────────────────────


def _do_apply():
    """Standalone --apply: overwrite original with v2 file."""
    if not OUTPUT_NEW.exists():
        logger.error("v2 parquet not found: %s", OUTPUT_NEW)
        sys.exit(1)
    import shutil

    shutil.copy2(OUTPUT_NEW, SENTIMENT_PARQUET)
    logger.info("Overwritten: %s", SENTIMENT_PARQUET)
    print("Original parquet replaced.")
    OUTPUT_NEW.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate sentiment parquet with cardiffnlp"
    )
    parser.add_argument("--test", action="store_true", help="Test on a small sample")
    parser.add_argument("--n", type=int, default=2000, help="Rows for test mode")
    parser.add_argument(
        "--apply", action="store_true", help="Replace original parquet with v2 file"
    )
    args = parser.parse_args()

    if not SENTIMENT_PARQUET.exists():
        logger.error("Not found: %s", SENTIMENT_PARQUET)
        sys.exit(1)

    # Standalone --apply mode
    if args.apply and not args.test:
        _do_apply()
        return

    # Snapshot old predictions
    df_old = pd.read_parquet(SENTIMENT_PARQUET)[
        ["comment_id", "text_clean", "language", "sentiment_bert"]
    ].copy()

    # Load full data, strip old sentiment columns
    df_full = pd.read_parquet(SENTIMENT_PARQUET)
    present = [c for c in SENTIMENT_COLS if c in df_full.columns]
    if present:
        df_full = df_full.drop(columns=present)
    logger.info("Base data: %d rows, %d columns", len(df_full), len(df_full.columns))

    if args.test:
        n = min(args.n, len(df_full))
        df_full = df_full.sample(n=n, random_state=42).reset_index(drop=True)
        logger.info("Test mode: sampling %d rows", n)

    regenerate(df_full, is_test=args.test, test_n=args.n if args.test else 0)

    progress = load_progress()
    n_chunks_done = len(progress.get("completed_chunks", []))
    if n_chunks_done == 0:
        logger.warning("No chunks were processed.")
        return

    df_result = merge_chunks(max(progress["completed_chunks"]) + 1)
    logger.info("Result merged: %d rows", len(df_result))

    if args.test:
        show_verification(df_old, df_result, n=20)
        print(f"\n{'='*60}")
        print(f"  TEST COMPLETE: {len(df_result)} rows processed")
        print(f"  Model: {_MODEL_VERSION}")
        print("  Original parquet NOT modified.")
        print(f"{'='*60}")
        return

    # Full run: assemble final parquet
    df_original = pd.read_parquet(SENTIMENT_PARQUET)
    for c in SENTIMENT_COLS:
        if c in df_original.columns:
            df_original = df_original.drop(columns=[c])

    merge_cols = ["comment_id"] + SENTIMENT_COLS
    df_result = df_result[merge_cols].drop_duplicates(subset="comment_id")
    df_final = df_original.merge(df_result, on="comment_id", how="left")
    _atomic_write(df_final, OUTPUT_NEW)
    logger.info(
        "Saved to %s (%d rows, %d cols)",
        OUTPUT_NEW,
        len(df_final),
        len(df_final.columns),
    )

    print(f"\n{'='*60}")
    print(f"  Regenerated parquet: {OUTPUT_NEW.name}")
    print(f"  Model: {_MODEL_VERSION}")
    print(f"  Rows: {len(df_final)}")
    print(f"  Original untouched: {SENTIMENT_PARQUET.name}")
    print("  Report with --apply to replace original.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
