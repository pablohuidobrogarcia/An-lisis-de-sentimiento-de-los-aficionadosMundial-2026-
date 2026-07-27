import io
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── Load raw & processed ────────────────────────────────────
raw = pd.read_parquet("data/raw/youtube_comments.parquet")
proc = pd.read_parquet(
    "data/processed/comentarios_sentimiento/comentarios_sentimiento.parquet"
)

raw_ids = set(raw["comment_id"])
proc_ids = set(proc["comment_id"])

pending_ids = raw_ids - proc_ids
print(f"Raw total:        {len(raw):>8,}")
print(f"Processed (sent): {len(proc):>8,}")
print(f"Pending (raw - processed): {len(pending_ids):>8,}")
print()

pending = raw[raw["comment_id"].isin(pending_ids)].copy()
print(f"Filas pendientes (por comment_id): {len(pending):>8,}")
print()

# ── 1. Breakdown by match_date ──────────────────────────────
print("=" * 60)
print("1. DESGLOSE POR MATCH_DATE (pendientes)")
print("=" * 60)

date_dist = pending["match_date"].value_counts().sort_index()
for d, c in date_dist.items():
    pct = c / len(pending) * 100
    print(f"  {str(d)[:25]:>25}: {c:>8,}  ({pct:>5.1f}%)")
print(f"  {'TOTAL':>25}: {len(pending):>8,}  (100%)")
print()

# ── 2. Check preprocessing columns ──────────────────────────
print("=" * 60)
print("2. COLUMNAS DE PREPROCESSING EN PENDIENTES")
print("=" * 60)

checks = ["text_clean", "language", "lang_confidence", "is_spam", "tokens"]
for col in checks:
    if col in pending.columns:
        n_null = pending[col].isna().sum()
        n_non_null = len(pending) - n_null
        if col in ["text_clean", "tokens"]:
            n_empty = (pending[col].astype(str).str.strip() == "").sum()
            print(
                f"  {col:>20}: {n_non_null:>8,} no nulos  |  {n_null:>8,} nulos  |  {n_empty:>8,} vacios"
            )
        else:
            print(f"  {col:>20}: {n_non_null:>8,} no nulos  |  {n_null:>8,} nulos")
    else:
        print(f"  {col:>20}: COLUMNA NO EXISTE")

print()

# ── 3. Also check in raw all columns available ──────────────
print("=" * 60)
print("3. COLUMNAS DISPONIBLES EN RAW")
print("=" * 60)
print(f"  Columnas: {list(raw.columns)}")
print()

# ── 4. Duplicates and spam ──────────────────────────────────
print("=" * 60)
print("4. DUPLICADOS CONOCIDOS")
print("=" * 60)

# Check for duplicate comment_ids in raw
dup_ids = raw[raw.duplicated(subset=["comment_id"], keep=False)]
n_dup = len(dup_ids) // 2 if len(dup_ids) > 0 else 0
print(f"  Comment_ids duplicados en raw: {len(dup_ids)} filas ({n_dup} grupos)")

# Check text_hash duplicates in pending
if "text_hash" in pending.columns:
    hash_dups = pending[pending.duplicated(subset=["text_hash"], keep=False)]
    print(f"  text_hash duplicados en pendientes: {len(hash_dups)} filas")

# ── 5. Final projection ─────────────────────────────────────
print()
print("=" * 60)
print("5. PROYECCION FINAL")
print("=" * 60)

# Pending by match_date groups
print(f"\n  Pendientes totales:             {len(pending):>8,}")
print(
    f"  De la final (19 jul):           {date_dist.get('2026-07-19 19:00:00+00:00', 0):>8,}"
)

# Count how many have text_clean already
has_text_clean = "text_clean" in pending.columns and pending["text_clean"].notna().sum()
print(f"  Pendientes CON text_clean:      {has_text_clean:>8,}")
print(f"  Pendientes SIN text_clean:      {len(pending) - has_text_clean:>8,}")

# Check is_spam
if "is_spam" in pending.columns:
    n_spam = pending["is_spam"].sum()
    print(f"  Marcados como spam:             {n_spam:>8,}")
else:
    print("  is_spam: columna no existe")
    n_spam = 0

# Estimate final count
# The original processed had 168,569 minus duplicates and spam
# Pending unique comment_ids should be added
estimated = len(proc) + len(pending) - n_spam - n_dup
print("\n  ESTIMACION FINAL:")
print(f"    Procesado actual:               {len(proc):>8,}")
print(f"    + Pendientes unicos:            {len(pending):>8,}")
print(f"    - Spam (en pendientes):         {n_spam:>8,}")
print(f"    - Duplicados estimados:         {n_dup:>8,}")
print("    ─────────────────────────────────────")
print(f"    TOTAL ESTIMADO:                 {estimated:>8,}")
