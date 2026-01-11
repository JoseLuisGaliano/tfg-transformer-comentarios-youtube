import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm.auto import tqdm


def main():
    # Cargar el dataset de test
    df = pd.read_csv("test.csv")
    texts = df["text"].astype(str).tolist()
    true_labels = df["label"].astype(str).tolist()

    # Definir las clases
    candidate_labels = [
        "tecnologia",
        "musica",
        "noticias",
        "deportes",
        "cine",
        "animales",
        "ciencia",
        "viajes",
        "vehiculos",
    ]

    # Seleccionar dispositivo
    device = 0 if torch.cuda.is_available() else -1

    # Crear el pipeline zero-shot
    classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        device=device,
    )

    # Inferencia por batches
    BATCH_SIZE = 1
    predicted_labels = []

    print(f"Número de ejemplos en test: {len(texts)}")
    print(f"Usando dispositivo: {'GPU' if device == 0 else 'CPU'}")
    print("Lanzando inferencia zero-shot...\n")

    for i in tqdm(range(0, len(texts), BATCH_SIZE),
                  desc="Zero-shot inference",
                  unit="batch"):
        batch_texts = texts[i : i + BATCH_SIZE]

        # pipeline
        outputs = classifier(
            batch_texts,
            candidate_labels,
            multi_label=False  # una sola clase por comentario
        )

        # Si hay un solo elemento, outputs podría ser un dict
        if isinstance(outputs, dict):
            outputs = [outputs]

        for out in outputs:
            # out["labels"] está ordenado por score descendente
            top_label = out["labels"][0]
            predicted_labels.append(top_label)

    assert len(predicted_labels) == len(true_labels), "Longitudes no coinciden"
    
    # Guardar predicciones en CSV
    df_preds = pd.DataFrame({
        "text": texts,
        "pred": predicted_labels,
        "label": true_labels,
    })
    df_preds.to_csv("predicciones_test_zeroshot.csv", index=False, encoding="utf-8")
    print("Predicciones guardadas en predicciones_test_zeroshot.csv")


    # Métricas
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1_macro = f1_score(true_labels, predicted_labels, average="macro")
    report = classification_report(true_labels, predicted_labels, output_dict=True)

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "classification_report": report,
    }

    # Guardar métricas en JSON
    with open("zero_shot_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nMétricas principales:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1 macro:  {f1_macro:.4f}")
    print("\nMétricas completas guardadas en zero_shot_metrics.json")

    # Matriz de confusión
    labels_order = candidate_labels  # orden fijo de las clases
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels_order)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(labels_order)))
    ax.set_yticks(np.arange(len(labels_order)))
    ax.set_xticklabels(labels_order, rotation=45, ha="right")
    ax.set_yticklabels(labels_order)

    ax.set_ylabel("Etiqueta real")
    ax.set_xlabel("Etiqueta predicha")
    ax.set_title("Matriz de confusión - Zero-shot mDeBERTa-v3-base-mnli-xnli")

    # Anotar los valores dentro de cada celda
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig("zero_shot_confusion_matrix.png", dpi=300)
    plt.close(fig)

    print("Matriz de confusión guardada en zero_shot_confusion_matrix.png")


if __name__ == "__main__":
    main()

