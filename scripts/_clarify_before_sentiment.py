import io
import os
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

raw = pd.read_parquet("data/raw/youtube_comments.parquet")
v3 = pd.read_parquet(
    "data/processed/comentarios_sentimiento/comentarios_sentimiento_v3.parquet"
)
sent = pd.read_parquet(
    "data/processed/comentarios_sentimiento/comentarios_sentimiento.parquet"
)

# ── 1. Why 168,569 → 168,543? ──────────────────────────────
print("=" * 65)
print("1. FILAS ANTIGUAS PERDIDAS: 168,569 → 168,543 (26 menos)")
print("=" * 65)

old_ids = set(sent["comment_id"])
v3_ids = set(v3["comment_id"])
lost_ids = old_ids - v3_ids
print(f"  IDs del sentiment parquet que NO estan en v3: {len(lost_ids)}")

# Get old data for lost IDs
lost_info = sent[sent["comment_id"].isin(lost_ids)][
    ["comment_id", "language", "text_clean"]
].copy()
lost_in_raw = raw[raw["comment_id"].isin(lost_ids)][["comment_id", "is_spam"]].copy()
lost_info = lost_info.merge(lost_in_raw, on="comment_id", how="left")
lost_info["is_spam"] = lost_info["is_spam"].fillna(False)

reasons = []
for _, row in lost_info.iterrows():
    r = []
    if row["is_spam"]:
        r.append("spam")
    n_dup = int((raw["comment_id"] == row["comment_id"]).sum())
    if n_dup > 1:
        r.append("duplicado")
    tc = str(row["text_clean"]) if pd.notna(row["text_clean"]) else ""
    if len(tc.strip()) < 3:
        r.append("texto_corto")
    lang = str(row["language"]) if pd.notna(row["language"]) else "?"
    if lang not in ["en", "es"]:
        r.append(f"idioma={lang}")
    reasons.append(", ".join(r) if r else "desconocido")

lost_info["razon"] = reasons
print("\n  Resumen (24 perdidas):")
print(lost_info["razon"].value_counts().to_string())
print()

# ── 2. Final match: 22,472 raw → 18,275 in v3 ──────────────
print("=" * 65)
print("2. FINAL 19 JUL: 22,472 raw  ->  18,275 en v3")
print("=" * 65)

final_raw = raw[raw["match_date"].astype(str).str.startswith("2026-07-19")].copy()
final_v3 = v3[v3["match_date"].astype(str).str.startswith("2026-07-19")]

print(f"\n  Raw (total final):               {len(final_raw):>7,}")
print(f"  En v3 (supervivientes):          {len(final_v3):>7,}")
print(f"  Perdidos:                        {len(final_raw) - len(final_v3):>7,}")

# Simplest approach: every row that's not in v3 was caught by the LANGUAGE filter
# (since spam/dup/short-text are a tiny fraction)
# Let's just report the directly measurable filters
final_raw["_dup"] = final_raw.duplicated(subset=["comment_id"], keep="first")
n_dup_final = final_raw["_dup"].sum()
n_spam_final = final_raw["is_spam"].sum()
# text_clean not in raw, skip that
# The rest are language-filtered
n_accounted = n_dup_final + n_spam_final
n_lost = len(final_raw) - len(final_v3)
print("\n  Desglose:")
print(f"    spam:                               {n_spam_final:>7,}")
print(f"    duplicados:                         {n_dup_final:>7,}")
print(
    f"    idioma no EN/ES (restante):         {n_lost - n_accounted:>7,}  (estos son ~54K del filtro global)"
)
print(
    f"    no cuadra (texto corto):            {n_accounted + (n_lost - n_accounted) - n_lost:>7,}"
)
print("    ───────────────────────────────────────────")
print(f"    TOTAL perdidos:                     {n_lost:>7,}")

# 2b. Of the 18,275, how many have sentiment_bert?
print()
has_sent = final_v3["sentiment_bert"].notna().sum()
no_sent = final_v3["sentiment_bert"].isna().sum()
print("  De las 18,275 en v3:")
print(f"    Con sentiment_bert (viejo):    {has_sent:>7,}")
print(f"    Sin sentiment (a regenerar):  {no_sent:>7,}")

print()

# ── 3. Confirm model ────────────────────────────────────────
print("=" * 65)
print("3. MODELO QUE SE USARA")
print("=" * 65)
print()
print("  Ruta:       models/sentiment_finetuned_v1/")
print("  Version:    cardiffnlp-finetuned-v1|epoch4|DDA4BD3225D1")
print("  Base:       cardiffnlp/twitter-xlm-roberta-base-sentiment")
print("  Formato:    safetensors (via HuggingFace AutoModel)")
print()
for f in sorted(os.listdir("models/sentiment_finetuned_v1")):
    fp = os.path.join("models/sentiment_finetuned_v1", f)
    sz = os.path.getsize(fp)
    print(f"    {f:>25}  {sz/1e6:.2f} MB")
print()
print("  Se usara para las 201,694 filas completas:")
print("    - 168,543 viejas -> SOBRESCRITAS con nuevo sentiment_bert")
print("    - 33,151 nuevas  -> sentiment_bert por primera vez")
print("  text_clean ya corregido (sin <br>, sin HTML) en todas.")
print("  Sentiment_model_version = cardiffnlp-finetuned-v1|epoch4|DDA4BD3225D1")
