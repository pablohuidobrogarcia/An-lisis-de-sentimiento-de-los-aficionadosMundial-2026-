"""
Compare alternative sentiment models against manual labels.

Models:
  1. nlptown/bert-base-multilingual-uncased-sentiment (current, from CSV)
  2. pysentimiento (robertuito ES / twitter-roberta EN)
  3. cardiffnlp/twitter-xlm-roberta-base-sentiment (multilingual)

Usage:
    python scripts/compare_sentiment_models.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

DATA_PATH = _project_root / "evaluation" / "manual_labels_random_sample.csv"

# ── Load & prepare data ──────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df = df.dropna(subset=["manual_label", "text_clean"])
df["manual_label"] = df["manual_label"].str.strip().str.upper()
df["language"] = df["language"].str.strip().str.lower()

VALID = {"POS", "NEG", "NEU"}
df = df[df["manual_label"].isin(VALID)].reset_index(drop=True)

y_true = df["manual_label"]
texts = df["text_clean"].tolist()
langs = df["language"].tolist()
n = len(df)

print(f"Loaded {n} labeled samples\n")

# ── Helper: evaluate & print per-class metrics ──────────────────────────────


def evaluate(name: str, y_true: pd.Series, y_pred: list, inf_time: float):
    y_pred_s = pd.Series(y_pred)
    mask = y_pred_s.isin(VALID)
    yt, yp = y_true[mask], y_pred_s[mask]

    acc = accuracy_score(yt, yp)
    report = classification_report(yt, yp, labels=["POS", "NEG", "NEU"], digits=3)
    cm = confusion_matrix(yt, yp, labels=["POS", "NEG", "NEU"])

    prec = precision_score(yt, yp, labels=["POS", "NEG", "NEU"], average=None)
    rec = recall_score(yt, yp, labels=["POS", "NEG", "NEU"], average=None)
    f1 = f1_score(yt, yp, labels=["POS", "NEG", "NEU"], average=None)

    return {
        "name": name,
        "accuracy": acc,
        "precision_pos": prec[0],
        "precision_neg": prec[1],
        "precision_neu": prec[2],
        "recall_pos": rec[0],
        "recall_neg": rec[1],
        "recall_neu": rec[2],
        "f1_pos": f1[0],
        "f1_neg": f1[1],
        "f1_neu": f1[2],
        "confusion_matrix": cm,
        "report": report,
        "time": inf_time,
        "n_valid": len(yt),
    }


# ── 1. Current model: nlptown (from CSV column) ─────────────────────────────
start = time.time()
y_nlptown = df["sentiment_bert_predicted"].str.strip().str.upper().tolist()
# nlptown mapping done at generation time; column is already POS/NEG/NEU
time_nlptown = time.time() - start

m_nlptown = evaluate("nlptown (current)", y_true, y_nlptown, time_nlptown)

# ── 2. pysentimiento ─────────────────────────────────────────────────────────
print("Loading pysentimiento analyzers ...")
from pysentimiento import create_analyzer

analyzer_es = create_analyzer(task="sentiment", lang="es")
analyzer_en = create_analyzer(task="sentiment", lang="en")

start = time.time()
preds_pysentimiento = []
for text, lang in zip(texts, langs):
    try:
        if lang == "es":
            result = analyzer_es.predict(text)
        else:
            result = analyzer_en.predict(text)
        preds_pysentimiento.append(result.output)
    except Exception:
        preds_pysentimiento.append("NEU")
time_pysentimiento = time.time() - start

m_pysentimiento = evaluate(
    "pysentimiento", y_true, preds_pysentimiento, time_pysentimiento
)

# ── 3. cardiffnlp/twitter-xlm-roberta-base-sentiment ────────────────────────
print("Loading cardiffnlp/twitter-xlm-roberta-base-sentiment pipeline ...")
from transformers import pipeline

pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    max_length=512,
    truncation=True,
)

# Inspect label mapping
id2label = pipe.model.config.id2label
print(f"  Model label mapping: {id2label}")

label_map = {}
for k, v in id2label.items():
    label_map[v.lower()] = v.upper()[:3]

start = time.time()
preds_cardiff = []
for text in texts:
    try:
        result = pipe(text)[0]
        label = label_map.get(result["label"].lower(), "NEU")
        preds_cardiff.append(label)
    except Exception:
        preds_cardiff.append("NEU")
time_cardiff = time.time() - start

m_cardiff = evaluate("cardiffnlp/xlm-roberta", y_true, preds_cardiff, time_cardiff)

# ── Print per-model detail ───────────────────────────────────────────────────
for m in [m_nlptown, m_pysentimiento, m_cardiff]:
    print(f"\n{'='*60}")
    print(f"  {m['name']}")
    print(f"{'='*60}")
    print(f"  Accuracy:       {m['accuracy']:.3f} ({m['accuracy']:.1%})")
    print(f"  Valid samples:  {m['n_valid']}/{n}")
    print(f"  Inference time: {m['time']:.2f}s")
    print("\n  Per-class metrics:")
    print("                  POS     NEG     NEU")
    print(
        f"  Precision       {m['precision_pos']:.3f}   {m['precision_neg']:.3f}   {m['precision_neu']:.3f}"
    )
    print(
        f"  Recall          {m['recall_pos']:.3f}   {m['recall_neg']:.3f}   {m['recall_neu']:.3f}"
    )
    print(
        f"  F1              {m['f1_pos']:.3f}   {m['f1_neg']:.3f}   {m['f1_neu']:.3f}"
    )
    print("\n  Confusion Matrix:")
    print("          Pred POS  Pred NEG  Pred NEU")
    cm = m["confusion_matrix"]
    print(f"  True POS  {cm[0,0]:>3d}       {cm[0,1]:>3d}       {cm[0,2]:>3d}")
    print(f"  True NEG  {cm[1,0]:>3d}       {cm[1,1]:>3d}       {cm[1,2]:>3d}")
    print(f"  True NEU  {cm[2,0]:>3d}       {cm[2,1]:>3d}       {cm[2,2]:>3d}")

# ── Final comparison table ───────────────────────────────────────────────────
print(f"\n{'='*90}")
print("  FINAL COMPARISON TABLE")
print(f"{'='*90}")
print(
    f"  {'Modelo':<30s} {'Accuracy':>9s} {'Prec NEG':>9s} {'Recall NEG':>10s} {'F1 NEG':>8s} {'Tiempo':>8s}"
)
print(f"  {'-'*30} {'-'*9} {'-'*9} {'-'*10} {'-'*8} {'-'*8}")

for m in [m_nlptown, m_pysentimiento, m_cardiff]:
    print(
        f"  {m['name']:<30s} {m['accuracy']:>8.1%}  "
        f"{m['precision_neg']:>8.3f}  "
        f"{m['recall_neg']:>9.3f}  "
        f"{m['f1_neg']:>7.3f}  "
        f"{m['time']:>6.2f}s"
    )

print("\nNota: Tiempo de inferencia incluye carga de modelo para pysentimiento")
print("      y cardiffnlp (la primera ejecución descarga los pesos).")
