import os

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

base = r"C:\Users\Pablo\Desktop\Proyecto Mundial\An-lisis-de-sentimiento-de-los-aficionadosMundial-2026-\data\eval"

test_raw = pd.read_excel(os.path.join(base, "sentiment_labeling_test.xlsx"))
pred_test = pd.read_csv(os.path.join(base, "model_predictions_for_finetuning_test.csv"))

test_clean = test_raw.copy()
test_clean["manual_label"] = (
    test_clean["manual_label"].astype(str).str.strip().str.upper()
)

test_merged = test_clean.merge(pred_test, on="comment_id", how="inner")

y_true = test_merged["manual_label"].values
y_pred = test_merged["sentiment_bert"].values

labels = ["NEG", "NEU", "POS"]

print("=" * 70)
print("PASO 2 — EVALUACION DEL MODELO vs TEST SET HUMANO")
print("=" * 70)
print("Modelo: cardiffnlp/twitter-xlm-roberta-base-sentiment")
print("Test set: {} filas (etiquetado humano real)".format(len(test_merged)))
print()

# Accuracy
acc = accuracy_score(y_true, y_pred)
print("ACCURACY: {:.4f} ({:.2f}%)".format(acc, acc * 100))

# Cohen's kappa
kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
print("COHEN'S KAPPA: {:.4f}".format(kappa))
print()

# Precision, Recall, F1 per class
prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels)

print("METRICAS POR CLASE:")
print(
    "{:<6} {:>10} {:>10} {:>10} {:>10}".format(
        "Clase", "Precision", "Recall", "F1", "Support"
    )
)
print("-" * 50)
for i, lbl in enumerate(labels):
    print(
        "{:<6} {:>10.4f} {:>10.4f} {:>10.4f} {:>10d}".format(
            lbl, prec[i], rec[i], f1[i], support[i]
        )
    )

# Macro / Weighted avg
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=labels, average="macro"
)
prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=labels, average="weighted"
)
print()
print(
    "{:<6} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        "MACRO", prec_macro, rec_macro, f1_macro
    )
)
print(
    "{:<6} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        "WEIGHTED", prec_weighted, rec_weighted, f1_weighted
    )
)
print()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)
print("MATRIZ DE CONFUSION:")
print("{:<8}".format(""), end="")
for lbl in labels:
    print("{:>8}".format("Pred " + lbl), end="")
    print("{:>8}".format(""), end="")
print()
for i, lbl in enumerate(labels):
    print("{:<8}".format("Real " + lbl), end="")
    for j in range(len(labels)):
        print("{:>8d}    ".format(cm[i, j]), end="")
    print()

print()
print("Matriz raw:")
print("{:>8} {:>8} {:>8}".format("NEG", "NEU", "POS"))
print("NEG {}".format(cm[0]))
print("NEU {}".format(cm[1]))
print("POS {}".format(cm[2]))
