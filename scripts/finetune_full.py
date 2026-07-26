"""
Full fine-tuning: cardiffnlp/twitter-xlm-roberta-base-sentiment
- max_epochs=8, early stopping patience=2 on macro-F1
- Checkpoint por epoch con resume capability
- Partial FT (embeddings + 8 layers frozen, last 4 + classifier trainable)
- Class-weighted loss
- Test set untouched hasta evaluacion final
"""

import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import logging as hf_logging

sys.stdout.reconfigure(encoding="utf-8")
hf_logging.set_verbosity_error()

# ── Paths ────────────────────────────────────────────────────
BASE = r"C:\Users\Pablo\Desktop\Proyecto Mundial\An-lisis-de-sentimiento-de-los-aficionadosMundial-2026-"
DATA_EVAL = os.path.join(BASE, "data", "eval")
OUTPUT_DIR = os.path.join(DATA_EVAL, "finetune_output")
LOG_DIR = os.path.join(BASE, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, "finetune_sentiment.log"), encoding="utf-8"
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
BATCH_SIZE = 8
MAX_LEN = 128
LR = 3e-5
MAX_EPOCHS = 8
PATIENCE = 2
FREEZE_LAYERS = True
LAYERS_TO_UNFREEZE = 4  # last 4 encoder layers + classifier head

ID2LABEL = {0: "NEG", 1: "NEU", 2: "POS"}
LABEL2ID = {"NEG": 0, "NEU": 1, "POS": 2}

DEVICE = torch.device("cpu")


# ═══════════════════════════════════════════════════════════════
#  DATOS
# ═══════════════════════════════════════════════════════════════
def load_and_split():
    log.info("Cargando datos...")
    train_raw = pd.read_excel(os.path.join(DATA_EVAL, "sentiment_labeling_train.xlsx"))
    pred_train = pd.read_csv(
        os.path.join(DATA_EVAL, "model_predictions_for_finetuning_train.csv")
    )

    train_clean = train_raw.dropna(subset=["manual_label"]).copy()
    train_clean["manual_label"] = (
        train_clean["manual_label"].astype(str).str.strip().str.upper()
    )
    train_clean["label"] = train_clean["manual_label"].map(LABEL2ID)
    train_merged = train_clean.merge(pred_train, on="comment_id", how="inner")
    log.info(f"  Train valido: {len(train_merged)} filas")

    y = train_merged["label"].values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(sss.split(train_merged, y))
    train_df = train_merged.iloc[train_idx].reset_index(drop=True)
    val_df = train_merged.iloc[val_idx].reset_index(drop=True)
    log.info(f"  Split: train_final={len(train_df)}, validation={len(val_df)}")
    log.info(f'  Train dist: {train_df["manual_label"].value_counts().to_dict()}')
    log.info(f'  Val   dist: {val_df["manual_label"].value_counts().to_dict()}')

    return train_df, val_df


def prepare_loaders(train_df, val_df, tokenizer):
    train_texts = train_df["text_clean"].tolist()
    train_labels = train_df["label"].tolist()
    val_texts = val_df["text_clean"].tolist()
    val_labels = val_df["label"].tolist()

    train_enc = tokenizer(
        train_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    val_enc = tokenizer(
        val_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    train_dataset = torch.utils.data.TensorDataset(
        train_enc["input_ids"],
        train_enc["attention_mask"],
        torch.tensor(train_labels, dtype=torch.long),
    )
    val_dataset = torch.utils.data.TensorDataset(
        val_enc["input_ids"],
        val_enc["attention_mask"],
        torch.tensor(val_labels, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, train_labels


# ═══════════════════════════════════════════════════════════════
#  MODELO
# ═══════════════════════════════════════════════════════════════
def create_model():
    log.info(f"Cargando modelo: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(DEVICE)

    if FREEZE_LAYERS:
        for param in model.roberta.embeddings.parameters():
            param.requires_grad = False
        frozen_layers = 12 - LAYERS_TO_UNFREEZE
        for param in model.roberta.encoder.layer[:-LAYERS_TO_UNFREEZE].parameters():
            param.requires_grad = False
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"  Partial FT: embeddings + {frozen_layers} layers frozen")
        log.info(
            f"  Parametros: {total:,} total, {trainable:,} trainable ({100*trainable/total:.1f}%)"
        )
    else:
        log.info("  Full FT: todas las capas entrenables")

    return model


# ═══════════════════════════════════════════════════════════════
#  RESUME
# ═══════════════════════════════════════════════════════════════
def get_resume_state():
    state_path = os.path.join(OUTPUT_DIR, "training_state.json")
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
        last_epoch = state.get("last_completed_epoch", 0)
        best_mf1 = state.get("best_macro_f1", 0.0)
        best_neu_f1 = state.get("best_neu_f1", 0.0)
        patience_counter = state.get("patience_counter", 0)
        # Verificar que el checkpoint existe
        ckpt_path = os.path.join(OUTPUT_DIR, f"epoch_{last_epoch}_complete.pt")
        if os.path.exists(ckpt_path) and last_epoch > 0:
            log.info(f"Resume detectado: reanudando desde epoch {last_epoch}")
            return last_epoch, best_mf1, best_neu_f1, patience_counter, ckpt_path
    return 0, 0.0, 0.0, 0, None


def save_epoch_checkpoint(epoch, model, optimizer, scheduler, train_loss, metrics):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "metrics": metrics,
    }
    path = os.path.join(OUTPUT_DIR, f"epoch_{epoch}_complete.pt")
    torch.save(ckpt, path)
    # Clean up previous checkpoint
    prev_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch-1}_complete.pt")
    if os.path.exists(prev_path):
        os.remove(prev_path)
    # Save training state metadata
    state = {
        "last_completed_epoch": int(epoch),
        "best_macro_f1": float(metrics.get("best_macro_f1", 0.0)),
        "best_neu_f1": float(metrics.get("best_neu_f1", 0.0)),
        "patience_counter": int(metrics.get("patience_counter", 0)),
    }
    with open(os.path.join(OUTPUT_DIR, "training_state.json"), "w") as f:
        json.dump(state, f)
    return path


def save_best_checkpoint(epoch, model, macro_f1, neu_f1):
    path = os.path.join(OUTPUT_DIR, "best_model.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "macro_f1": macro_f1,
            "neu_f1": neu_f1,
        },
        path,
    )
    # Also save for easy HuggingFace loading
    model.save_pretrained(os.path.join(OUTPUT_DIR, "best_model_hf"))
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.save_pretrained(os.path.join(OUTPUT_DIR, "best_model_hf"))
    log.info(
        f"  [*] Best model saved (epoch {epoch}, macro_f1={macro_f1:.4f}, neu_f1={neu_f1:.4f})"
    )


# ═══════════════════════════════════════════════════════════════
#  EVAL
# ═══════════════════════════════════════════════════════════════
def evaluate(model, val_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "f1_neg": f1_per_class[0],
        "f1_neu": f1_per_class[1],
        "f1_pos": f1_per_class[2],
        "predictions": all_preds,
        "labels": all_labels,
    }


# ═══════════════════════════════════════════════════════════════
#  FINAL TEST EVAL
# ═══════════════════════════════════════════════════════════════
def evaluate_on_test():
    log.info("")
    log.info("=" * 60)
    log.info("EVALUACION FINAL — TEST SET (500 held-out)")
    log.info("=" * 60)

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

    model = create_model()
    ckpt = torch.load(
        os.path.join(OUTPUT_DIR, "best_model.pt"),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    log.info(
        f'  Best checkpoint loaded (epoch {ckpt["epoch"]}, macro_f1={ckpt["macro_f1"]:.4f})'
    )

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

    results = evaluate(model, test_loader)
    y_true = results["labels"]
    y_pred = results["predictions"]

    log.info(
        f'  Accuracy:       {results["accuracy"]:.4f} ({results["accuracy"]*100:.2f}%)'
    )
    log.info(f'  Macro-F1:       {results["macro_f1"]:.4f}')
    kappa = cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2])
    log.info(f"  Cohen's Kappa:  {kappa:.4f}")
    log.info("")
    log.info("  F1 per class:")
    log.info(f'    NEG: {results["f1_neg"]:.4f}')
    log.info(f'    NEU: {results["f1_neu"]:.4f}')
    log.info(f'    POS: {results["f1_pos"]:.4f}')
    log.info("")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    log.info("  Confusion Matrix:")
    log.info("           Pred NEG  Pred NEU  Pred POS")
    log.info(f"  Real NEG  {cm[0][0]:5d}     {cm[0][1]:5d}     {cm[0][2]:5d}")
    log.info(f"  Real NEU  {cm[1][0]:5d}     {cm[1][1]:5d}     {cm[1][2]:5d}")
    log.info(f"  Real POS  {cm[2][0]:5d}     {cm[2][1]:5d}     {cm[2][2]:5d}")

    # Baseline comparison
    baseline_pred = test_merged["sentiment_bert"].map(LABEL2ID).values
    baseline_f1_macro = f1_score(y_true, baseline_pred, average="macro")
    baseline_f1_neu = f1_score(y_true, baseline_pred, average=None)[1]
    log.info("")
    log.info("  COMPARATIVA vs BASELINE (modelo original):")
    log.info(
        f'    Macro-F1: {results["macro_f1"]:.4f} (finetune) vs {baseline_f1_macro:.4f} (baseline)'
    )
    log.info(
        f'    NEU F1:   {results["f1_neu"]:.4f} (finetune) vs {baseline_f1_neu:.4f} (baseline)'
    )


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    log.info("")
    log.info("#" * 60)
    log.info(f"FINE-TUNING — {MODEL_NAME}")
    log.info(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    log.info(
        f"Config: max_epochs={MAX_EPOCHS}, patience={PATIENCE}, batch={BATCH_SIZE}, lr={LR}"
    )
    log.info(f"Device: {DEVICE}")
    log.info("#" * 60)

    # ── Resume check ─────────────────────────────────────────
    (
        start_epoch,
        best_macro_f1,
        best_neu_f1,
        patience_counter,
        resume_ckpt,
    ) = get_resume_state()
    if start_epoch >= MAX_EPOCHS:
        log.info(
            f"Entrenamiento ya completado ({start_epoch} epochs). Saltando a evaluacion final."
        )
        evaluate_on_test()
        return

    # ── Load data ────────────────────────────────────────────
    train_df, val_df = load_and_split()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, val_loader, train_labels = prepare_loaders(
        train_df, val_df, tokenizer
    )

    # ── Model ────────────────────────────────────────────────
    model = create_model()

    # ── Resume weights ───────────────────────────────────────
    if resume_ckpt is not None:
        ckpt_data = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt_data["model_state_dict"])
        log.info(f"  Weights restored from {resume_ckpt}")

    # ── Class weights ────────────────────────────────────────
    class_counts = np.bincount(train_labels, minlength=3)
    weights = 1.0 / class_counts.astype(float)
    weights = weights / weights.sum() * len(class_counts)
    log.info(f"  Class weights (NEG,NEU,POS): {np.round(weights, 3)}")
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    )

    # ── Optimizer ────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ── Restore opt/sched if resuming ────────────────────────
    if resume_ckpt is not None and "optimizer_state_dict" in ckpt_data:
        optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])

    # ── Training loop ───────────────────────────────────────
    log.info("")
    log.info(f"Iniciando entrenamiento (epochs {start_epoch+1} a {MAX_EPOCHS})")
    log.info(
        f'{"Epoch":>6} | {"Loss":>6} | {"Time":>7} | {"Val Acc":>7} | {"Macro-F1":>8} | {"NEU F1":>6} | {"Patience":>8}'
    )
    log.info("-" * 65)

    training_start = time.time()

    for epoch in range(start_epoch + 1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids, attention_mask, labels = [x.to(DEVICE) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - training_start

        # Validation
        val_results = evaluate(model, val_loader)
        macro_f1 = val_results["macro_f1"]
        neu_f1 = val_results["f1_neu"]

        # Early stopping
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_neu_f1 = neu_f1
            patience_counter = 0
            save_best_checkpoint(epoch, model, macro_f1, neu_f1)
        else:
            patience_counter += 1

        metrics = {
            "best_macro_f1": best_macro_f1,
            "best_neu_f1": best_neu_f1,
            "patience_counter": patience_counter,
            "macro_f1": macro_f1,
            "neu_f1": neu_f1,
            "accuracy": val_results["accuracy"],
            "epoch_time": epoch_time,
            "elapsed": elapsed,
        }

        # Save epoch checkpoint
        save_epoch_checkpoint(epoch, model, optimizer, scheduler, avg_loss, metrics)

        log.info(
            f"{epoch:>6d} | {avg_loss:>6.4f} | {epoch_time:>6.0f}s | "
            f'{val_results["accuracy"]:>7.4f} | {macro_f1:>8.4f} | {neu_f1:>6.4f} | {patience_counter:>4d}/{PATIENCE}'
        )

        if patience_counter >= PATIENCE:
            log.info("")
            log.info(
                f"Early stopping triggered after epoch {epoch} "
                f"(no macro-F1 improvement for {PATIENCE} epochs)"
            )
            break

    # ── Summary ──────────────────────────────────────────────
    total_time = time.time() - training_start
    best_ckpt = torch.load(
        os.path.join(OUTPUT_DIR, "best_model.pt"),
        map_location="cpu",
        weights_only=False,
    )
    best_epoch = best_ckpt["epoch"]

    log.info("")
    log.info("=" * 60)
    log.info("ENTRENAMIENTO COMPLETADO")
    log.info("=" * 60)
    log.info(f"Total epochs completed: {epoch}")
    log.info(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    log.info(f"Best macro-F1: {best_macro_f1:.4f} (epoch {best_epoch})")
    log.info(f"Best NEU F1: {best_neu_f1:.4f}")

    log.info(f"Best model restored from epoch {best_epoch} for final evaluation")

    # ── Final test evaluation ────────────────────────────────
    evaluate_on_test()

    log.info("")
    log.info(f'Finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    log.info("Finetune completo. Resultados en logs/finetune_sentiment.log")


if __name__ == "__main__":
    main()
