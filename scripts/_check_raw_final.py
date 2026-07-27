import io
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

df = pd.read_parquet("data/raw/youtube_comments.parquet")

print(f"Total filas: {len(df)}")
print(f"Rango collected_at: {df['collected_at'].min()} a {df['collected_at'].max()}")

# Final match comments (July 19)
date_col = df["match_date"].astype(str)
mask = date_col.str.startswith("2026-07-19")
print(f"\nComentarios final (19 jul): {mask.sum()}")
if mask.any():
    sub = df[mask]
    print(
        f"  Rango collected_at: {sub['collected_at'].min()} a {sub['collected_at'].max()}"
    )
    print(
        f"  Rango published_at: {sub['published_at'].min()} a {sub['published_at'].max()}"
    )

# Latest collected_at
print("\nUltimas collected_at por dia:")
counts = df["collected_at"].value_counts().sort_index()
print(counts.tail(5).to_string())

print("\nTotal filas Spain+Argentina (final):")
final_teams = df[
    df["teams"].apply(lambda x: "Spain" in str(x) and "Argentina" in str(x))
]
print(f"  {len(final_teams)}")
