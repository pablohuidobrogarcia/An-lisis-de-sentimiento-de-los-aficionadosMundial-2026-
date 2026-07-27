import io
import os
import sys

import pandas as pd

# Add project root to path so we can import src
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.preprocessing import clean_text

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── Load data ───────────────────────────────────────────────
raw = pd.read_parquet("data/raw/youtube_comments.parquet")
sent = pd.read_parquet(
    "data/processed/comentarios_sentimiento/comentarios_sentimiento.parquet"
)

# Merge old sentiment with raw text
old = sent[["comment_id", "text_clean"]].merge(
    raw[["comment_id", "text"]], on="comment_id", how="inner"
)
print(f"Filas old con raw text disponible: {len(old):>8,}")
print(f"  (de {len(sent):>8,} en sentiment parquet)")
print()

# ── Regenerate text_clean from raw text ─────────────────────
print("Regenerando text_clean desde text original...")
old["text_clean_new"] = old["text"].apply(clean_text)
print("  Hecho.")
print()

# ── Compare ─────────────────────────────────────────────────
old["changed"] = old["text_clean"] != old["text_clean_new"]
n_changed = old["changed"].sum()
n_unchanged = len(old) - n_changed

print("=" * 60)
print(f"IMPACTO: {n_changed:,} filas CAMBIAN / {n_unchanged:,} sin cambios")
print("=" * 60)
print(f"  Total filas verificadas: {len(old):,}")
print(f"  Cambiaron:               {n_changed:>8,}  ({n_changed/len(old)*100:.2f}%)")
print(
    f"  Sin cambios:             {n_unchanged:>8,}  ({n_unchanged/len(old)*100:.2f}%)"
)
print()

# ── Before/after examples ───────────────────────────────────
print("=" * 60)
print("EJEMPLOS ANTES / DESPUES (10 filas)")
print("=" * 60)

examples = old[old["changed"]].head(10)
for i, (_, row) in enumerate(examples.iterrows()):
    old_txt = row["text_clean"]
    new_txt = row["text_clean_new"]
    print(f"\n--- Ejemplo {i+1} ---")
    print(f"  RAW:      {row['text'][:150]}")
    print(f"  ANTES:    {old_txt[:150]}")
    print(f"  DESPUES:  {new_txt[:150]}")
    if old_txt != new_txt:
        print(f"  DIFERENCIA: +{len(new_txt)-len(old_txt)} chars")

print()
print("=" * 60)
print("VERIFICACION: filas que NO cambiaron (spot-check)")
print("=" * 60)
unchanged_samples = old[~old["changed"]].sample(3)
for i, (_, row) in enumerate(unchanged_samples.iterrows()):
    print(f"\n--- Sin cambio {i+1} ---")
    print(f"  RAW:      {row['text'][:120]}")
    print(f"  text_clean: {row['text_clean'][:120]}")

print()
print("=" * 60)
print("VERIFICACION: emojis preservados")
print("=" * 60)
emoji_rows = old[
    old["text_clean_new"].str.contains(r"[\U0001F600-\U0001F9FF]", na=False)
]
print(f"  Filas con emojis en text_clean_new: {len(emoji_rows):,}")
if len(emoji_rows) > 0:
    print(f"  Ejemplo: {emoji_rows.iloc[0]['text_clean_new'][:120]}")
