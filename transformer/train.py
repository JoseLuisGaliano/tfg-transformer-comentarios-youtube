import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch

import matplotlib
matplotlib.use("Agg")  # backend no interactivo para guardar figuras
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset, DatasetDict


def build_label_mapping(series: pd.Series) -> Dict[str, int]:
    unique_labels = sorted(series.unique())
    return {lbl: i for i, lbl in enumerate(unique_labels)}


def encode_labels(series: pd.Series, label2id: Dict[str, int]) -> np.ndarray:
    return series.map(label2id).astype(int).values


@dataclass
class WeightedTrainer(Trainer):
    """Permite usar class weights si se proporcionan."""
    class_weights: Optional[torch.Tensor] = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1m}


def plot_confusion_matrix(cm, classes, out_path: str, title="Matriz de confusión"):
    num_classes = len(classes)
    fig_size = max(6, min(0.6 * num_classes, 20))
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # Normalizamos por filas para facilitar lectura (opcional)
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, cm_sum, where=cm_sum != 0)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]}\n({cm_norm[i, j]:.2f})"
            plt.text(j, i, txt,
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=8,
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("Etiqueta real")
    plt.xlabel("Predicción")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_roc_curves(y_true, y_score, classes, out_path: str):
    """
    y_true: array shape (N,) con ids de clase
    y_score: array shape (N, C) con probabilidades (softmax)
    """
    n_classes = y_score.shape[1]
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    # AUC micro
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    # AUC macro (promedio de curvas por clase)
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}
    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

    # Interpolación para macro-average
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes
    auc_macro = auc(all_fpr, mean_tpr)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_micro, tpr_micro, label=f"micro-average ROC (AUC = {auc_micro:.3f})")
    plt.plot(all_fpr, mean_tpr, label=f"macro-average ROC (AUC = {auc_macro:.3f})")

    # Por clase
    for i in range(n_classes):
        plt.plot(fpr_dict[i], tpr_dict[i], lw=1, alpha=0.6,
                 label=f"Clase {classes[i]} (AUC = {auc_dict[i]:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Curvas ROC")
    plt.legend(loc="lower right", fontsize=8, ncol=1 if n_classes <= 10 else 2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "roc_auc_micro": float(auc_micro),
        "roc_auc_macro": float(auc_macro),
        **{f"roc_auc_class_{classes[i]}": float(auc_dict[i]) for i in range(n_classes)}
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True, help="Ruta al CSV de entrenamiento")
    parser.add_argument("--val_csv", type=str, required=True, help="Ruta al CSV de validación")
    parser.add_argument("--test_csv", type=str, required=True, help="Ruta al CSV de test")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")

    parser.add_argument("--model_name_or_path", type=str, default="distilbert-base-multilingual-cased",
                        help="Modelo base.")
    parser.add_argument("--output_dir", type=str, default="outputs/model_distil")
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")

    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_class_weights", action="store_true",
                        help="Si se activa, calcula pesos de clase inversos a la frecuencia.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Usando GPU: {device_name}")
    else:
        print("No se detecta GPU, se usará CPU.")
    
    # Reproducibilidad
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1) Cargando CSVs ya separados
    print("Cargando CSV…")
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    # Filtrado mínimo de filas inválidas
    for df in [train_df, val_df, test_df]:
        df[args.text_col] = df[args.text_col].astype(str).str.strip()
    train_df = train_df.dropna(subset=[args.text_col, args.label_col])
    val_df = val_df.dropna(subset=[args.text_col, args.label_col])
    test_df = test_df.dropna(subset=[args.text_col, args.label_col])
    train_df = train_df[train_df[args.text_col] != ""]
    val_df = val_df[val_df[args.text_col] != ""]
    test_df = test_df[test_df[args.text_col] != ""]

    # 2) Split por grupos (video_id)
    # ya hecho

    # 3) Label mapping (string -> id)
    print("Mapeando labels...")
    label2id = build_label_mapping(train_df[args.label_col])
    id2label = {v: k for k, v in label2id.items()}

    # Verifica que val/test no contengan etiquetas desconocidas
    unknown_val = set(val_df[args.label_col].unique()) - set(label2id.keys())
    unknown_test = set(test_df[args.label_col].unique()) - set(label2id.keys())
    if unknown_val or unknown_test:
        raise ValueError(f"Se encontraron etiquetas desconocidas en val/test. "
                         f"VAL desconocidas: {unknown_val}, TEST desconocidas: {unknown_test}")

    # 4) Añadir columna numeric label
    train_df["label_id"] = encode_labels(train_df[args.label_col], label2id)
    val_df["label_id"] = encode_labels(val_df[args.label_col], label2id)
    test_df["label_id"] = encode_labels(test_df[args.label_col], label2id)

    # 5) Tokenizer y datasets
    print("Tokenizando...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    def tok_fn(batch):
        return tokenizer(
            batch[args.text_col],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    train_ds = Dataset.from_pandas(train_df[[args.text_col, "label_id"]].rename(columns={"label_id": "labels"}))
    val_ds = Dataset.from_pandas(val_df[[args.text_col, "label_id"]].rename(columns={"label_id": "labels"}))
    test_ds = Dataset.from_pandas(test_df[[args.text_col, "label_id"]].rename(columns={"label_id": "labels"}))

    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=[args.text_col])
    val_ds = val_ds.map(tok_fn, batched=True, remove_columns=[args.text_col])
    test_ds = test_ds.map(tok_fn, batched=True, remove_columns=[args.text_col])

    datasets = DatasetDict(train=train_ds, validation=val_ds, test=test_ds)

    # 6) Config y modelo
    num_labels = len(label2id)
    hf_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=hf_config
    )

    # 7) Opcional: class weights contra desbalance
    class_weights_tensor = None
    if args.use_class_weights:
        # Pesos inversamente proporcionales a la frecuencia en TRAIN
        counts = train_df["label_id"].value_counts().sort_index()
        inv_freq = 1.0 / counts
        class_weights = (inv_freq / inv_freq.sum()) * len(inv_freq)  # normaliza pero mantiene proporciones
        class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float)
        print("Class weights:", class_weights_tensor.tolist())

    # 8) Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 9) Training args (con varias defensas anti-overfitting)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,

        max_grad_norm=args.max_grad_norm,
        label_smoothing_factor=args.label_smoothing,

        fp16=args.fp16,
        bf16=args.bf16,

        seed=args.seed,
        dataloader_num_workers=4,
        report_to="none",
    )

    # 10) Trainer (+ early stopping)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]

    trainer_cls = WeightedTrainer if args.use_class_weights else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
        **({"class_weights": class_weights_tensor} if args.use_class_weights else {})
    )

    # 11) Entrenar
    print("ENTRENANDO!")
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 12) Evaluar
    print("Evaluando...")
    metrics_val = trainer.evaluate(datasets["validation"])
    print("Validation:", metrics_val)
    metrics_test = trainer.evaluate(datasets["test"])
    print("Test:", metrics_test)
    
    # 13) Matriz de confusión y ROC-AUC en TEST
    print("Generando métricas...")
    # Obtener logits en test
    test_pred = trainer.predict(datasets["test"])
    test_logits = test_pred.predictions
    test_labels = test_pred.label_ids
    test_preds = np.argmax(test_logits, axis=-1)

    # Softmax para probabilidades
    probs = torch.softmax(torch.tensor(test_logits), dim=1).cpu().numpy()

    # Matriz de confusión
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(num_labels)))
    # Guardar CSV
    cm_csv_path = os.path.join(args.output_dir, "confusion_matrix.csv")
    pd.DataFrame(cm, index=[id2label[i] for i in range(num_labels)],
                    columns=[id2label[i] for i in range(num_labels)]
                 ).to_csv(cm_csv_path, encoding="utf-8")
    # Guardar PNG
    cm_png_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, classes=[id2label[i] for i in range(num_labels)],
                          out_path=cm_png_path,
                          title="Matriz de confusión (TEST)")

    # Curvas ROC-AUC (micro/macro y por clase)
    roc_png_path = os.path.join(args.output_dir, "roc_curves.png")
    auc_dict = plot_roc_curves(
        y_true=test_labels,
        y_score=probs,
        classes=[id2label[i] for i in range(num_labels)],
        out_path=roc_png_path
    )

    # AUC micro/macro a fichero
    with open(os.path.join(args.output_dir, "roc_auc.json"), "w", encoding="utf-8") as f:
        json.dump(auc_dict, f, ensure_ascii=False, indent=2)

    # Añadir AUC al print final
    print("ROC-AUC (TEST):", {"micro": auc_dict["roc_auc_micro"], "macro": auc_dict["roc_auc_macro"]})


    # 13) Guardar métricas y label mapping
    with open(os.path.join(args.output_dir, "train_results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "train": {k: float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v
                          for k, v in train_result.metrics.items()},
                "validation": {k: float(v) for k, v in metrics_val.items()},
                "test": {k: float(v) for k, v in metrics_test.items()},
            },
            f, ensure_ascii=False, indent=2
        )

    with open(os.path.join(args.output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    # 14) Info final
    print("\n=== Resumen ===")
    print(f"Salida: {args.output_dir}")
    print(f"Num labels: {num_labels}")
    print(f"Ejemplos -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    print(f"Mejor modelo cargado al final según: {training_args.metric_for_best_model}")


if __name__ == "__main__":
    main()

