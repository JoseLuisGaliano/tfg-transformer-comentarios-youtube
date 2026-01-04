import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import shap
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)


def load_model_and_tokenizer(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    device = 0 if torch.cuda.is_available() else -1

    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-classification",
        return_all_scores=True,
        device=device,
    )

    id2label = getattr(model.config, "id2label", {}) or {}
    num_labels = model.config.num_labels

    label_names = []
    for i in range(num_labels):
        if i in id2label:              # caso claves int
            label_names.append(id2label[i])
        elif str(i) in id2label:       # caso claves str
            label_names.append(id2label[str(i)])
        else:
            # fallback por si falta algo
            label_names.append(f"label_{i}")

    print("Clases detectadas en el modelo:", label_names)

    return pipe, tokenizer, label_names


def compute_shap_values(pipe, tokenizer, texts, label_names):
    # Masker para texto
    masker = shap.maskers.Text(tokenizer)

    # Explainer model-agnostic sobre el pipeline de HF
    explainer = shap.Explainer(
        pipe,
        masker=masker,
        output_names=label_names,
    )
    
    # SHAP sobre los textos (puede tardar si son muchos)
    explanation = explainer(texts)
    
    return explanation


def aggregate_token_importances(explanation, num_classes, top_k=20):
    # Diccionarios {token: [valores_shap]} por clase
    token_shap_values = [defaultdict(list) for _ in range(num_classes)]

    # values: lista de arrays (n_tokens_i, n_classes)
    values_list = explanation.values
    tokens_list = explanation.data

    n_samples = len(values_list)

    for i in range(n_samples):
        sample_values = np.array(values_list[i])   # (n_tokens_i, n_outputs) o (n_tokens_i,)
        sample_tokens = tokens_list[i]             # lista de longitud n_tokens_i

        # Normalizar dimensiones: (n_tokens,) -> (n_tokens, 1)
        if sample_values.ndim == 1:
            sample_values = sample_values[:, np.newaxis]

        n_tokens, n_outputs = sample_values.shape
        max_classes = min(num_classes, n_outputs)

        # Por si hay desajuste raro entre tokens y SHAPs
        n_tokens = min(n_tokens, len(sample_tokens))

        for j in range(n_tokens):
            tok = sample_tokens[j]
            if tok is None:
                continue

            tok_str = str(tok).strip()
            if tok_str == "":
                continue

            # Ignorar tokens especiales típicos
            if tok_str in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"]:
                continue

            for class_idx in range(max_classes):
                val = float(sample_values[j, class_idx])
                token_shap_values[class_idx][tok_str].append(val)

    # Media por token y clase + top_k positivos/negativos
    token_mean_shap = []
    top_tokens = []

    for class_idx in range(num_classes):
        mean_dict = {
            tok: float(np.mean(vals))
            for tok, vals in token_shap_values[class_idx].items()
            if len(vals) > 0
        }
        token_mean_shap.append(mean_dict)

        if mean_dict:
            sorted_tokens_pos = sorted(
                mean_dict.items(), key=lambda x: x[1], reverse=True
            )[:top_k]
            sorted_tokens_neg = sorted(
                mean_dict.items(), key=lambda x: x[1]
            )[:top_k]
        else:
            sorted_tokens_pos, sorted_tokens_neg = [], []

        top_tokens.append((sorted_tokens_pos, sorted_tokens_neg))

    return token_mean_shap, top_tokens


def plot_global_token_importances(
    top_tokens, label_names, output_dir, top_k=20
):
    os.makedirs(output_dir, exist_ok=True)

    num_classes = len(label_names)

    for class_idx in range(num_classes):
        label = label_names[class_idx]
        pos_tokens, neg_tokens = top_tokens[class_idx]

        # Positivos
        if pos_tokens:
            tokens_pos = [t for t, v in pos_tokens]
            values_pos = [v for t, v in pos_tokens]

            plt.figure(figsize=(10, 6))
            y_pos = np.arange(len(tokens_pos))
            plt.barh(y_pos, values_pos)
            plt.yticks(y_pos, tokens_pos)
            plt.xlabel("Media valor SHAP (contribución positiva)")
            plt.title(f"Top {top_k} tokens positivos para clase '{label}'")
            plt.gca().invert_yaxis()
            plt.tight_layout()

            fname_pos = os.path.join(
                output_dir, f"global_tokens_{label}_positivos.png"
            )
            plt.savefig(fname_pos, dpi=200)
            plt.close()

        # Negativos
        if neg_tokens:
            tokens_neg = [t for t, v in neg_tokens]
            values_neg = [v for t, v in neg_tokens]

            plt.figure(figsize=(10, 6))
            y_neg = np.arange(len(tokens_neg))
            plt.barh(y_neg, values_neg)
            plt.yticks(y_neg, tokens_neg)
            plt.xlabel("Media valor SHAP (contribución negativa)")
            plt.title(f"Top {top_k} tokens negativos para clase '{label}'")
            plt.gca().invert_yaxis()
            plt.tight_layout()

            fname_neg = os.path.join(
                output_dir, f"global_tokens_{label}_negativos.png"
            )
            plt.savefig(fname_neg, dpi=200)
            plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="model_distil", help="Carpeta del modelo")
    parser.add_argument("--test_csv", type=str, default="test.csv", help="Ruta al test.csv con al menos la columna 'text'.")
    parser.add_argument("--output_dir", type=str, default="shap_global_plots", help="Carpeta de salida para los gráficos.")
    parser.add_argument("--top_k", type=int, default=20, help="Número de tokens positivos/negativos a mostrar por clase.")

    args = parser.parse_args()

    # Cargar modelo, tokenizer y nombres de clases
    pipe, tokenizer, label_names = load_model_and_tokenizer(args.model_dir)
    num_classes = len(label_names)

    # Cargar datos de test
    df = pd.read_csv(args.test_csv)
    if "text" not in df.columns:
        raise ValueError("El csv debe contener una columna 'text'.")

    texts = df["text"].astype(str).tolist()
    
    n = len(texts) 
    if n == 0:
        raise ValueError("El csv de test está vacío.")
    print(f"Calculando SHAP sobre {n} textos...")

    # Calcular valores SHAP
    explanation = compute_shap_values(
        pipe,
        tokenizer,
        texts,
        label_names
    )

    # Agregar por token y clase
    print("Agregando contribuciones por token y clase...")
    token_mean_shap, top_tokens = aggregate_token_importances(
        explanation, num_classes=num_classes, top_k=args.top_k
    )

    # Guardar gráficos
    print(f"Generando gráficos en '{args.output_dir}'...")
    plot_global_token_importances(
        top_tokens,
        label_names,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )

    print("Listo.")


if __name__ == "__main__":
    main()

