import io
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.preprocessing import SUPPORTED_LANGUAGES, clean_text, detect_language

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── Load data ───────────────────────────────────────────────
print("Cargando datos...")
raw = pd.read_parquet("data/raw/youtube_comments.parquet")
sent = pd.read_parquet(
    "data/processed/comentarios_sentimiento/comentarios_sentimiento.parquet"
)
print(f"  Raw: {len(raw):,} filas")
print(f"  Sent: {len(sent):,} filas")
print()

# ── 1. Regenerate text_clean for ALL rows ───────────────────
print("Paso 1/4: Regenerando text_clean desde text original (todas las filas)...")
raw["text_clean"] = raw["text"].apply(clean_text)
n_clean = raw["text_clean"].notna().sum()
print(f"  Hecho. text_clean generado para {n_clean:,} filas")
print()

# ── 2. Carry old sentiment columns first ────────────────────
print("Paso 2/4: Fusionando sentiment_bert del parquet antiguo...")
sent_old = sent[
    [
        "comment_id",
        "sentiment_bert",
        "sentiment_bert_probas",
        "sentiment_baseline",
        "sentiment_model_version",
        "language",
        "lang_confidence",
    ]
].drop_duplicates(subset=["comment_id"])
raw = raw.merge(sent_old, on="comment_id", how="left", suffixes=("", "_old"))

# For old rows, use language/confidence from sentiment parquet (already exists)
# For new rows, detect now
print("  Detectando language en filas NUEVAS...")
new_mask = raw["language"].isna()


def detect_lang(text):
    try:
        code, conf = detect_language(text)
        return code, conf
    except Exception:
        return None, None


if new_mask.any():
    new_texts = raw.loc[new_mask, "text_clean"]
    lang_results = new_texts.apply(lambda x: detect_lang(x))
    raw.loc[new_mask, "language"] = lang_results.apply(lambda x: x[0])
    raw.loc[new_mask, "lang_confidence"] = lang_results.apply(lambda x: x[1])

n_with = raw["sentiment_bert"].notna().sum()
n_without = len(raw) - n_with
print(f"  Con sentiment_bert (old): {n_with:,} | Sin (new): {n_without:,}")
print(
    f"  Language: EN={(raw['language']=='en').sum():,} | ES={(raw['language']=='es').sum():,}"
)
print()

# ── 3. Filter and dedup ─────────────────────────────────────
print("Paso 3/4: Filtrando spam y duplicados...")

n_spam = raw["is_spam"].sum()
raw = raw[~raw["is_spam"]]
print(f"  Eliminados {n_spam} spam")

n_before = len(raw)
raw = raw.drop_duplicates(subset=["comment_id"], keep="first")
n_dups = n_before - len(raw)
print(f"  Eliminados {n_dups} duplicados por comment_id")

raw = raw[raw["text_clean"].notna() & (raw["text_clean"].str.len() >= 3)]
n_short = n_before - n_dups - len(raw)
print(f"  Eliminadas {n_short} filas por text_clean muy corto")

n_lang_before = len(raw)
raw = raw[raw["language"].isin(SUPPORTED_LANGUAGES)]
n_dropped_lang = n_lang_before - len(raw)
print(f"  Eliminadas {n_dropped_lang} filas por language no soportado")

print(f"\n  TOTAL FINAL: {len(raw):,} filas")
print()

# ── 4. Summary + Save ───────────────────────────────────────
print("=" * 60)
print("RESUMEN DATASET UNIFICADO (pre-sentimiento)")
print("=" * 60)
print(f"  Total filas:               {len(raw):>8,}")
print(f"  Con sentiment_bert (old):  {raw['sentiment_bert'].notna().sum():>8,}")
print(f"  Sin sentiment_bert (new):  {raw['sentiment_bert'].isna().sum():>8,}")
print(
    f"  De la final (19 jul):      {(raw['match_date'].astype(str).str.startswith('2026-07-19')).sum():>8,}"
)
print(f"  Language EN:               {(raw['language']=='en').sum():>8,}")
print(f"  Language ES:               {(raw['language']=='es').sum():>8,}")
print()

output_path = (
    "data/processed/comentarios_sentimiento/comentarios_sentimiento_v3.parquet"
)
raw.to_parquet(output_path, index=False)
print(f"Guardado en: {output_path}")
print("Done.")
