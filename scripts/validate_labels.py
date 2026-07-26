import os

import pandas as pd

base = r"C:\Users\Pablo\Desktop\Proyecto Mundial\An-lisis-de-sentimiento-de-los-aficionadosMundial-2026-\data\eval"

train_raw = pd.read_excel(os.path.join(base, "sentiment_labeling_train.xlsx"))
test_raw = pd.read_excel(os.path.join(base, "sentiment_labeling_test.xlsx"))

pred_train = pd.read_csv(
    os.path.join(base, "model_predictions_for_finetuning_train.csv")
)
pred_test = pd.read_csv(os.path.join(base, "model_predictions_for_finetuning_test.csv"))

print("=" * 70)
print("PASO 1 — CONSOLIDACIÓN Y VALIDACIÓN")
print("=" * 70)

print("\n--- 1.2 TRAIN: sentiment_labeling_train.xlsx ---")
print("Filas crudas: {}".format(len(train_raw)))
dups = train_raw["comment_id"].duplicated().sum()
print("Duplicados de comment_id: {}".format(dups))
print("comment_id únicos: {}".format(train_raw["comment_id"].nunique()))

empty_label = train_raw["manual_label"].isna() | (
    train_raw["manual_label"].astype(str).str.strip() == ""
)
print("Filas con manual_label vacío (serán excluidas): {}".format(empty_label.sum()))

label_vals = (
    train_raw.loc[~empty_label, "manual_label"].astype(str).str.strip().value_counts()
)
print("Valores únicos de manual_label (no vacíos):")
print(label_vals.to_string())

print("\n--- 1.3 TEST: sentiment_labeling_test.xlsx ---")
print("Filas crudas: {}".format(len(test_raw)))
empty_label_test = test_raw["manual_label"].isna() | (
    test_raw["manual_label"].astype(str).str.strip() == ""
)
print("Filas con manual_label vacío: {}".format(empty_label_test.sum()))
label_vals_test = (
    test_raw.loc[~empty_label_test, "manual_label"]
    .astype(str)
    .str.strip()
    .value_counts()
)
print("Valores únicos de manual_label (no vacíos):")
print(label_vals_test.to_string())

print("\n--- 1.4 CRUCE CON PREDICCIONES ---")

train_clean = train_raw[~empty_label].copy()
train_clean["manual_label"] = (
    train_clean["manual_label"].astype(str).str.strip().str.upper()
)

label_map = {
    "POS": "POS",
    "POSITIVE": "POS",
    "POSITIVO": "POS",
    "NEG": "NEG",
    "NEGATIVE": "NEG",
    "NEGATIVO": "NEG",
    "NEU": "NEU",
    "NEUTRAL": "NEU",
    "NEUTRO": "NEU",
}
train_clean["manual_label"] = (
    train_clean["manual_label"].map(label_map).fillna(train_clean["manual_label"])
)

bad_labels = train_clean[~train_clean["manual_label"].isin(["POS", "NEU", "NEG"])]
if len(bad_labels) > 0:
    print("[!] Labels anomalas en TRAIN despues de normalizar:")
    print(bad_labels[["comment_id", "manual_label"]].to_string())
else:
    print("[OK] TRAIN: todas las labels son POS/NEU/NEG despues de normalizar")

test_clean = test_raw[~empty_label_test].copy()
test_clean["manual_label"] = (
    test_clean["manual_label"].astype(str).str.strip().str.upper()
)
test_clean["manual_label"] = (
    test_clean["manual_label"].map(label_map).fillna(test_clean["manual_label"])
)
bad_labels_test = test_clean[~test_clean["manual_label"].isin(["POS", "NEU", "NEG"])]
if len(bad_labels_test) > 0:
    print("[!] Labels anomalas en TEST despues de normalizar:")
    print(bad_labels_test[["comment_id", "manual_label"]].to_string())
else:
    print("[OK] TEST: todas las labels son POS/NEU/NEG despues de normalizar")

train_merged = train_clean.merge(pred_train, on="comment_id", how="inner")
test_merged = test_clean.merge(pred_test, on="comment_id", how="inner")

print(
    "\nTrain después de merge con predicciones: {} filas (de {} limpias)".format(
        len(train_merged), len(train_clean)
    )
)
print(
    "Test  después de merge con predicciones: {} filas (de {} limpias)".format(
        len(test_merged), len(test_clean)
    )
)

if len(train_merged) < len(train_clean):
    missing = set(train_clean["comment_id"]) - set(pred_train["comment_id"])
    print(
        "  [!] comment_id de train sin match en predicciones: {}".format(len(missing))
    )

print("\n" + "=" * 70)
print("RESUMEN FINAL - PASO 1")
print("=" * 70)

print("\nTRAIN: {} filas validas (de 2800 originales)".format(len(train_merged)))
train_cls = train_merged["manual_label"].value_counts()
for lbl in ["POS", "NEU", "NEG"]:
    c = train_cls.get(lbl, 0)
    print("  {}: {} ({:.1f}%)".format(lbl, c, c / len(train_merged) * 100))

print("\nTEST: {} filas validas (de 500 originales)".format(len(test_merged)))
test_cls = test_merged["manual_label"].value_counts()
for lbl in ["POS", "NEU", "NEG"]:
    c = test_cls.get(lbl, 0)
    print("  {}: {} ({:.1f}%)".format(lbl, c, c / len(test_merged) * 100))

print("\nDistribucion de idiomas en TRAIN:")
train_lang = train_merged["language"].value_counts()
for lang, c in train_lang.items():
    print("  {}: {} ({:.1f}%)".format(lang, c, c / len(train_merged) * 100))

print("\nDistribucion de idiomas en TEST:")
test_lang = test_merged["language"].value_counts()
for lang, c in test_lang.items():
    print("  {}: {} ({:.1f}%)".format(lang, c, c / len(test_merged) * 100))

print("\nIdioma vs Label en TRAIN:")
ct = pd.crosstab(train_merged["language"], train_merged["manual_label"])
print(ct.to_string())

print("\nIdioma vs Label en TEST:")
ct2 = pd.crosstab(test_merged["language"], test_merged["manual_label"])
print(ct2.to_string())
