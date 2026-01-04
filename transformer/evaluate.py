import argparse
import json
import os
from typing import List

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
import torch.nn.functional as F

def load_label_mapping(model_dir: str):
    path = os.path.join(model_dir, "label_mapping.json")
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    id2label = {int(k): v for k, v in mapping.get("id2label", {}).items()}
    label2id = {k: int(v) for k, v in mapping.get("label2id", {}).items()}
    return id2label, label2id


def build_label_mapping_from_data(df, label_col: str):
    unique_labels = sorted(df[label_col].unique())
    label2id = {lbl: i for i, lbl in enumerate(unique_labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return id2label, label2id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directorio con el modelo entrenado")
    parser.add_argument("--label_col", type=str, default="label", help="Nombre de la columna con las etiquetas")
    parser.add_argument("--test_csv", type=str, default="test.csv", help="Ruta al CSV de test")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_csv", type=str, default="predicciones_test_transformer.csv", help="CSV de salida con predicciones")
    args = parser.parse_args()

    # Carga modelo/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Lee CSV
    df = pd.read_csv(args.test_csv, index_col=False, header=0)

    # Usa directamente la columna 'text' como entrada
    df["input_text"] = df["text"].fillna("").astype(str)
    
    # Intenta cargar label_mapping.json, si no existe lo reconstruimos
    id2label, label2id = load_label_mapping(args.model_dir)
    if label2id is None:
        print("No se encontró label_mapping.json, reconstruyendo desde el CSV de test...")
        id2label, label2id = build_label_mapping_from_data(df, args.label_col)
        print(f"Mapeado reconstruido: {len(label2id)} clases")
        print(f"  Clases: {list(label2id.keys())}")
        
    # Remapeo de etiquetas según el modelo
    df["label_idx"] = df[args.label_col].map(lambda c: label2id.get(str(c), -1))

    # Filtrar filas con etiqueta desconocida (-1)
    eval_df = df[df["label_idx"] >= 0].reset_index(drop=True)
    skipped = len(df) - len(eval_df)
    if len(eval_df) == 0:
        raise ValueError("No hay filas evaluables (todas las etiquetas son desconocidas para el modelo).")

    texts: List[str] = eval_df["input_text"].tolist()
    labels = torch.tensor(eval_df["label_idx"].tolist(), dtype=torch.long)

    # Recolectar todas las predicciones
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i : i + args.batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            
            # Calcular probabilidades con softmax
            probs = F.softmax(logits, dim=-1).cpu()
            preds = torch.argmax(logits, dim=-1).cpu()
            batch_labels = labels[i : i + args.batch_size]

            all_preds.extend(preds.tolist())
            all_labels.extend(batch_labels.tolist())
            all_probs.extend(probs.tolist())

    # Convertir a arrays de numpy para sklearn
    all_preds = pd.Series(all_preds)
    all_labels = pd.Series(all_labels)

    # Preparar nombres de clases para el reporte
    if id2label is not None:
        target_names = [id2label[i] for i in sorted(id2label.keys())]
    else:
        target_names = [str(i) for i in sorted(all_labels.unique())]

    # Generar reporte de clasificación completo
    print("" + "="*70)
    print("REPORTE DE EVALUACIÓN")
    print("="*70 + "")

    if skipped > 0:
        print(f"Se ignoraron {skipped} filas con etiquetas no vistas durante el entrenamiento")

    # Imprimir reporte detallado
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    print(report)

    # Calcular accuracy por separado para mostrarla al final
    accuracy = accuracy_score(all_labels, all_preds)
    print("="*70)
    print(f"ACCURACY FINAL: {accuracy:.4f}")
    print("="*70)
    
    # Crear DataFrame con predicciones
    predictions_df = pd.DataFrame({
        'text': eval_df['text'],
        'probabilidades': all_probs,
        'pred': [id2label[p] if id2label else str(p) for p in all_preds],
        'label': eval_df[args.label_col]
    })

    # Guardar CSV
    predictions_df.to_csv(args.output_csv, index=False)
    print(f"Predicciones guardadas en: {args.output_csv}")

if __name__ == "__main__":
    main()
