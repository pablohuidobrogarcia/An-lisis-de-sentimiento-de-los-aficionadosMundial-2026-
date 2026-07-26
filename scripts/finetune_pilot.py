"""
PILOTO: Fine-tuning de cardiffnlp/twitter-xlm-roberta-base-sentiment
- 300 muestras de train_final, 1 epoch
- Mide tiempo, RAM, verifica loop completo
- Estrategia: partial freeze (últimas 2 capas + classifier head)
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import psutil
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import logging as hf_logging

# Forzar UTF-8 en stdout (evita errores cp1252 con caracteres Unicode)
sys.stdout.reconfigure(encoding="utf-8")

# ── Config ──────────────────────────────────────────────────
BASE = r"C:\Users\Pablo\Desktop\Proyecto Mundial\An-lisis-de-sentimiento-de-los-aficionadosMundial-2026-\data\eval"
OUTPUT_DIR = os.path.join(BASE, "finetune_pilot_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
BATCH_SIZE = 8
MAX_LEN = 128
LR = 3e-5
NUM_EPOCHS = 1
PILOT_SIZE = 300
FREEZE_LAYERS = True  # Partial FT: congelar primeras N capas
LAYERS_TO_UNFREEZE = (
    4  # Últimas N capas + classifier head (freeze embeddings + primeras 8)
)

ID2LABEL = {0: "NEG", 1: "NEU", 2: "POS"}
LABEL2ID = {"NEG": 0, "NEU": 1, "POS": 2}

# Silenciar warnings de transformers
hf_logging.set_verbosity_error()


# ── RAM tracking ────────────────────────────────────────────
def log_ram(tag):
    proc = psutil.Process()
    mem = proc.memory_info()
    rss_mb = mem.rss / 1e6
    avail = psutil.virtual_memory().available / 1e6
    sys.stdout.write(f"  [RAM {tag}] RSS={rss_mb:.0f} MB | avail={avail:.0f} MB\n")
    sys.stdout.flush()
    return rss_mb


# ── Cargar datos ────────────────────────────────────────────
print("=" * 60)
print("PILOTO FINE-TUNING — cardiffnlp/twitter-xlm-roberta-base-sentiment")
print("=" * 60)

print("\n[1] Cargando datos...")
train_raw = pd.read_excel(os.path.join(BASE, "sentiment_labeling_train.xlsx"))
pred_train = pd.read_csv(
    os.path.join(BASE, "model_predictions_for_finetuning_train.csv")
)

# Limpiar
train_clean = train_raw.dropna(subset=["manual_label"]).copy()
train_clean["manual_label"] = (
    train_clean["manual_label"].astype(str).str.strip().str.upper()
)
train_clean["label"] = train_clean["manual_label"].map(LABEL2ID)

# Merge con predicciones (para mantener trazabilidad, aunque no las usamos para entrenar)
train_merged = train_clean.merge(pred_train, on="comment_id", how="inner")
print(f"  Filas totales válidas: {len(train_merged)}")

# ── Split estratificado ─────────────────────────────────────
print("\n[2] Split train_final / validation (90/10 estratificado)...")
y = train_merged["label"].values
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_idx, val_idx = next(sss.split(train_merged, y))

train_df = train_merged.iloc[train_idx].reset_index(drop=True)
val_df = train_merged.iloc[val_idx].reset_index(drop=True)

print(f"  train_final: {len(train_df)} | validation: {len(val_df)}")
print(f'  Train class dist: {train_df["manual_label"].value_counts().to_dict()}')
print(f'  Val   class dist: {val_df["manual_label"].value_counts().to_dict()}')

# ── Submuestreo para piloto ─────────────────────────────────
print(f"\n[3] Submuestreando {PILOT_SIZE} filas de train_final para piloto...")
pilot_df = (
    train_df.groupby("label", group_keys=False)
    .apply(
        lambda x: x.sample(
            max(1, int(PILOT_SIZE * len(x) / len(train_df))), random_state=42
        )
    )
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

# Asegurar exactamente PILOT_SIZE
if len(pilot_df) > PILOT_SIZE:
    pilot_df = pilot_df.iloc[:PILOT_SIZE]
elif len(pilot_df) < PILOT_SIZE:
    extra = train_df.drop(pilot_df.index).sample(
        PILOT_SIZE - len(pilot_df), random_state=42
    )
    pilot_df = pd.concat([pilot_df, extra]).reset_index(drop=True)

print(f"  Pilot train: {len(pilot_df)} | Val: {len(val_df)}")
print(f'  Pilot class dist: {pilot_df["manual_label"].value_counts().to_dict()}')

# ── Tokenizer ───────────────────────────────────────────────
print("\n[4] Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_fn(examples):
    return tokenizer(
        examples["text_clean"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )


pilot_ds = Dataset.from_pandas(pilot_df[["text_clean", "label"]])
val_ds = Dataset.from_pandas(val_df[["text_clean", "label"]])

pilot_ds = pilot_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)

pilot_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

pilot_loader = DataLoader(pilot_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ── Modelo ──────────────────────────────────────────────────
print(f"\n[5] Cargando modelo: {MODEL_NAME}")
device = torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)
model.to(device)
log_ram("after model load")

if FREEZE_LAYERS:
    frozen_layers = 12 - LAYERS_TO_UNFREEZE
    print(f"  Partial FT: congelando embeddings + primeras {frozen_layers} capas...")
    # Congelar embeddings
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
    # Congelar encoder layers excepto las últimas LAYERS_TO_UNFREEZE
    for name, param in model.roberta.encoder.layer[
        :-LAYERS_TO_UNFREEZE
    ].named_parameters():
        param.requires_grad = False
    # Contar parámetros entrenables
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parámetros totales: {total_params:,}")
    print(
        f"  Parámetros entrenables: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)"
    )
else:
    print("  Full fine-tuning (todas las capas)")

log_ram("after freeze")

# ── Class-weighted loss ─────────────────────────────────────
print("\n[6] Calculando pesos para class-weighted loss...")
class_counts = pilot_df["label"].value_counts().sort_index()
weights = 1.0 / class_counts.values.astype(float)
weights = weights / weights.sum() * len(class_counts)
print(f"  Pesos por clase (NEG, NEU, POS): {np.round(weights, 3)}")

# ── Optimizer + Scheduler ───────────────────────────────────
print("\n[7] Configurando optimizer y scheduler...")
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(pilot_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)

# ── Training loop ───────────────────────────────────────────
print(f"\n[8] ENTRENANDO — {NUM_EPOCHS} epoch(s), batch={BATCH_SIZE}, lr={LR}")
print("─" * 60)

log_ram("before training")
train_start = time.time()
loss_fn = torch.nn.CrossEntropyLoss(
    weight=torch.tensor(weights, dtype=torch.float32, device=device)
)

best_macro_f1 = 0.0
best_state = None

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    model.train()
    total_loss = 0

    for step, batch in enumerate(pilot_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if (step + 1) % 10 == 0:
            print(
                f"  epoch {epoch+1}/{NUM_EPOCHS} | step {step+1}/{len(pilot_loader)} | loss={loss.item():.4f}"
            )

    avg_loss = total_loss / len(pilot_loader)
    epoch_time = time.time() - epoch_start

    # ── Validación ──────────────────────────────────────────
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_per_class = f1_score(all_labels, all_preds, average=None)

    print(
        f"\n  >>> Epoch {epoch+1} | loss={avg_loss:.4f} | time={epoch_time:.1f}s | val_acc={acc:.4f} | val_macro_f1={f1_macro:.4f}"
    )
    print(f"      F1 per class (NEG,NEU,POS): {np.round(f1_per_class, 4)}")

    # Guardar mejor checkpoint por macro-F1
    if f1_macro > best_macro_f1:
        best_macro_f1 = f1_macro
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        ckpt_path = os.path.join(OUTPUT_DIR, "best_model_pilot.pt")
        torch.save(
            {"model_state_dict": best_state, "macro_f1": f1_macro, "epoch": epoch},
            ckpt_path,
        )
        print(f"      [*] Nuevo mejor checkpoint guardado (macro_f1={f1_macro:.4f})")

total_time = time.time() - train_start
log_ram("after training")

print("\n" + "=" * 60)
print("RESULTADOS DEL PILOTO")
print("=" * 60)
print(f"Tiempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"Tiempo por epoch: {total_time/NUM_EPOCHS:.1f}s")
print(f"Mejor macro-F1 en validation: {best_macro_f1:.4f}")
print(f"Checkpoint guardado en: {OUTPUT_DIR}")
print()

# ── Estimar full run ────────────────────────────────────────
samples_pilot = len(pilot_df)
samples_full = len(train_df)
ratio = samples_full / samples_pilot

print("── ESTIMACIÓN PARA FULL RUN ──")
print(f"Tamaño train_final completo: {samples_full}")
print(f"Factor de escala vs piloto: {ratio:.1f}x")
print()

for num_epochs in [3, 4, 5, 6]:
    est_time = total_time * ratio * num_epochs / NUM_EPOCHS
    hours = est_time / 3600
    mins = (est_time % 3600) / 60
    print(f"  {num_epochs} epochs → ~{est_time/60:.0f} min ({hours:.0f}h {mins:.0f}m)")
