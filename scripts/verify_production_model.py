"""
Verify that the production model at models/sentiment_finetuned_v1/
produces EXACTLY the same test metrics as the training checkpoint.
Loads from HuggingFace safetensors format (production path),
not from the .pt checkpoint used during training.
"""
import os
import sys

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

BASE = r"C:\Users\Pablo\Desktop\Proyecto Mundial\An-lisis-de-sentimiento-de-los-aficionadosMundial-2026-"
DATA_EVAL = os.path.join(BASE, "data", "eval")
PROD_MODEL_PATH = os.path.join(BASE, "models", "sentiment_finetuned_v1")
BATCH_SIZE = 8
MAX_LEN = 128
LABEL2ID = {"NEG": 0, "NEU": 1, "POS": 2}
ID2LABEL = {0: "NEG", 1: "NEU", 2: "POS"}
DEVICE = torch.device("cpu")

# ── Expected metrics (from successful training run) ──────────
EXPECTED = {
    "accuracy": 0.726,
    "macro_f1": 0.7101112117672767,
    "kappa": 0.5681584637788971,
    "f1_neg": 0.7682672233820459,
    "f1_neu": 0.5950413223140496,
    "f1_pos": 0.7670250896057348,
}

print("=" * 70)
print("VERIFICACION — Modelo produccion vs Test Set")
print("=" * 70)
print()
print(f"Cargando modelo desde: {PROD_MODEL_PATH}")
print(f"Dispositivo: {DEVICE}")
print()

# ── Load test set ────────────────────────────────────────────
test_raw = pd.read_excel(os.path.join(DATA_EVAL, "sentiment_labeling_test.xlsx"))
pred_test = pd.read_csv(
    os.path.join(DATA_EVAL, "model_predictions_for_finetuning_test.csv")
)
test_clean = test_raw.copy()
test_clean["manual_label"] = (
    test_clean["manual_label"].astype(str).str.strip().str.upper()
)
test_clean["label"] = test_clean["manual_label"].map(LABEL2ID)
test_merged = test_clean.merge(pred_test, on="comment_id", how="inner")
print(f"Muestras de test: {len(test_merged)}")
print()

# ── Load model from PRODUCTION path (safetensors) ────────────
print("Cargando modelo desde ruta de produccion...")
tokenizer = AutoTokenizer.from_pretrained(PROD_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    PROD_MODEL_PATH,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)
model.to(DEVICE)
model.eval()
print("  OK — modelo cargado desde safetensors")
print()

# ── Tokenize ─────────────────────────────────────────────────
test_enc = tokenizer(
    test_merged["text_clean"].tolist(),
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN,
    return_tensors="pt",
)
test_dataset = torch.utils.data.TensorDataset(
    test_enc["input_ids"],
    test_enc["attention_mask"],
    torch.tensor(test_merged["label"].values, dtype=torch.long),
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ── Evaluate ─────────────────────────────────────────────────
all_preds, all_labels = [], []
with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

y_true = all_labels
y_pred = all_preds
labels_list = [0, 1, 2]

acc = accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred, labels=labels_list)
prec, rec, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=labels_list
)
f1_macro = f1_score(y_true, y_pred, average="macro")
cm = confusion_matrix(y_true, y_pred, labels=labels_list)

print(f'{"Metric":>20} {"Obtenido":>12} {"Esperado":>12} {"Match?":>10}')
print("-" * 56)
metrics = {
    "Accuracy": (acc, EXPECTED["accuracy"]),
    "Macro-F1": (f1_macro, EXPECTED["macro_f1"]),
    "Kappa": (kappa, EXPECTED["kappa"]),
    "F1-NEG": (f1[0], EXPECTED["f1_neg"]),
    "F1-NEU": (f1[1], EXPECTED["f1_neu"]),
    "F1-POS": (f1[2], EXPECTED["f1_pos"]),
}
all_match = True
for name, (got, exp) in metrics.items():
    match = abs(got - exp) < 1e-4
    status = "PASS" if match else "FAIL"
    print(f"{name:>20} {got:>12.6f} {exp:>12.6f} {status:>10}")
    if not match:
        all_match = False

print()
print("Confusion Matrix (rows=real, cols=pred):")
print(f'{"":>8} {"NEG":>8} {"NEU":>8} {"POS":>8}')
print(f'{"NEG":>8} {cm[0][0]:>8d} {cm[0][1]:>8d} {cm[0][2]:>8d}')
print(f'{"NEU":>8} {cm[1][0]:>8d} {cm[1][1]:>8d} {cm[1][2]:>8d}')
print(f'{"POS":>8} {cm[2][0]:>8d} {cm[2][1]:>8d} {cm[2][2]:>8d}')
print()
if all_match:
    print("RESULTADO: TODAS LAS METRICAS COINCIDEN — modelo de produccion OK")
else:
    print("RESULTADO: HAY DISCREPANCIAS — revisar la copia del checkpoint")
print("Done.")
