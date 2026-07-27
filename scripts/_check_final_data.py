import io
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

df = pd.read_parquet(
    "data/processed/comentarios_sentimiento/comentarios_sentimiento.parquet"
)

# Todas las combinaciones Spain+Argentina
mask = df["teams"].apply(lambda x: "Spain" in str(x) and "Argentina" in str(x))
print(f"Filas con Spain+Argentina en teams: {mask.sum()}")

# Especificamente la final (match_date = 19 julio)
final_mask = (df["match_date"] == "2026-07-19") & (
    df["teams"].apply(lambda x: "Spain" in str(x) and "Argentina" in str(x))
)
print(f"Filas de la FINAL (match_date=2026-07-19): {final_mask.sum()}")

if final_mask.any():
    sub = df[final_mask]
    print(
        f"  Rango published_at: {sub['published_at'].min()} a {sub['published_at'].max()}"
    )
    print(
        f"  Rango collected_at: {sub['collected_at'].min()} a {sub['collected_at'].max()}"
    )
    print(f"  Sample text: {sub['text_clean'].iloc[0][:120]}")
else:
    print("  NO hay datos de la final en el parquet actual")
    # Ver si hay al menos algunos con match_date 19 julio
    any_19 = df["match_date"] == "2026-07-19"
    print(
        f"  Total comentarios con match_date=2026-07-19 (cualquier equipo): {any_19.sum()}"
    )

# Distribucion de fechas de partido para Spain+Argentina
print("\nDistribucion por match_date (Spain+Argentina):")
date_dist = df[mask]["match_date"].value_counts().sort_index()
for d, c in date_dist.items():
    print(f"  {d}: {c}")
