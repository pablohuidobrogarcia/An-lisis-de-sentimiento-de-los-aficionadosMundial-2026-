"""
Evaluacion final del modelo fine-tuned contra el test set (500 held-out).
Carga el mejor checkpoint y produce metricas completas.
"""
import json
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
OUTPUT_DIR = os.path.join(DATA_EVAL, "finetune_output")
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
BATCH_SIZE = 8
MAX_LEN = 128
LABEL2ID = {"NEG": 0, "NEU": 1, "POS": 2}
ID2LABEL = {0: "NEG", 1: "NEU", 2: "POS"}
DEVICE = torch.device("cpu")

# ── Load test set ────────────────────────────────────────────
print("=" * 65)
print("EVALUACION FINAL — TEST SET (500 held-out)")
print("=" * 65)

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
print(f"Test samples: {len(test_merged)}")

# ── Load best model ──────────────────────────────────────────
print("Loading best model checkpoint...")
ckpt = torch.load(
    os.path.join(OUTPUT_DIR, "best_model.pt"), map_location=DEVICE, weights_only=False
)
print(f'  Best epoch: {ckpt["epoch"]}')
print(f'  Val macro-F1: {ckpt["macro_f1"]:.4f}')
print(f'  Val NEU F1: {ckpt["neu_f1"]:.4f}')

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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

print()
print(f"Accuracy:      {acc:.4f} ({acc*100:.2f}%)")
print(f"Macro-F1:      {f1_macro:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print()
print(f'{"Class":>6} {"Precision":>10} {"Recall":>10} {"F1":>10} {"Support":>10}')
print("-" * 50)
for i, lbl in enumerate(["NEG", "NEU", "POS"]):
    print(f"{lbl:>6} {prec[i]:>10.4f} {rec[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10d}")
print()
print("Confusion Matrix (rows=real, cols=pred):")
print(f'{"":>8} {"NEG":>8} {"NEU":>8} {"POS":>8}')
print(f'{"NEG":>8} {cm[0][0]:>8d} {cm[0][1]:>8d} {cm[0][2]:>8d}')
print(f'{"NEU":>8} {cm[1][0]:>8d} {cm[1][1]:>8d} {cm[1][2]:>8d}')
print(f'{"POS":>8} {cm[2][0]:>8d} {cm[2][1]:>8d} {cm[2][2]:>8d}')

# ── Baseline comparison ──────────────────────────────────────
print()
print("─" * 65)
print("COMPARATIVA vs BASELINE (modelo original, no fine-tuned)")
print("─" * 65)

baseline_pred = test_merged["sentiment_bert"].map(LABEL2ID).values
baseline_acc = accuracy_score(y_true, baseline_pred)
baseline_f1_macro = f1_score(y_true, baseline_pred, average="macro")
baseline_f1_per_class = f1_score(y_true, baseline_pred, average=None)
baseline_kappa = cohen_kappa_score(y_true, baseline_pred, labels=labels_list)

print(f'{"Metric":>20} {"Baseline":>12} {"Fine-tuned":>12} {"Change":>12}')
print("-" * 58)
print(f'{"Accuracy":>20} {baseline_acc:>12.4f} {acc:>12.4f} {acc-baseline_acc:>+12.4f}')
print(
    f'{"Macro-F1":>20} {baseline_f1_macro:>12.4f} {f1_macro:>12.4f} {f1_macro-baseline_f1_macro:>+12.4f}'
)
print(
    f'{"Kappa":>20} {baseline_kappa:>12.4f} {kappa:>12.4f} {kappa-baseline_kappa:>+12.4f}'
)
print()
for i, lbl in enumerate(["NEG", "NEU", "POS"]):
    change = f1[i] - baseline_f1_per_class[i]
    print(
        f'{"F1-" + lbl:>20} {baseline_f1_per_class[i]:>12.4f} {f1[i]:>12.4f} {change:>+12.4f}'
    )

# Save results JSON
results = {
    "baseline": {
        "accuracy": float(baseline_acc),
        "macro_f1": float(baseline_f1_macro),
        "kappa": float(baseline_kappa),
        "f1_neg": float(baseline_f1_per_class[0]),
        "f1_neu": float(baseline_f1_per_class[1]),
        "f1_pos": float(baseline_f1_per_class[2]),
    },
    "finetuned": {
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "kappa": float(kappa),
        "f1_neg": float(f1[0]),
        "f1_neu": float(f1[1]),
        "f1_pos": float(f1[2]),
        "confusion_matrix": cm.tolist(),
        "best_epoch": int(ckpt["epoch"]),
        "val_macro_f1": float(ckpt["macro_f1"]),
    },
}
with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print()
print(f"Resultados guardados en {OUTPUT_DIR}/test_results.json")
print("Done.")
