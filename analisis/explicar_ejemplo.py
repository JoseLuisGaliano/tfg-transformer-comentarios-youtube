import argparse
import os

import torch
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
        if i in id2label:              # claves int
            label_names.append(id2label[i])
        elif str(i) in id2label:       # claves str
            label_names.append(id2label[str(i)])
        else:
            label_names.append(f"label_{i}")

    print("Clases detectadas en el modelo:", label_names)

    return pipe, tokenizer, label_names


def explain_text_with_shap(pipe, tokenizer, label_names, text: str):
    masker = shap.maskers.Text(tokenizer)

    explainer = shap.Explainer(
        pipe,
        masker=masker,
        output_names=label_names,
    )

    # SHAP sobre un único texto (lo pasamos como lista)
    explanation = explainer([text])

    return explanation


def save_explanation_html(explanation, output_html: str, class_idx: int | None = None):
    exp_single = explanation[0]


    base_vals = np.array(exp_single.base_values)
    vals = np.array(exp_single.values)

    n_classes = base_vals.shape[-1] if base_vals.ndim > 0 else 1

    if n_classes > 1:
        if class_idx is None:
            # Heurística: clase con mayor "output" aproximado = base + sum(shap)
            approx_outputs = base_vals + vals.sum(axis=0)
            class_idx = int(np.argmax(approx_outputs))
        if not (0 <= class_idx < n_classes):
            raise ValueError(f"class_idx fuera de rango: {class_idx} (n_classes={n_classes})")

        base_value = float(base_vals[class_idx])
        shap_values = vals[:, class_idx]
    else:
        base_value = float(base_vals)
        shap_values = vals

    features = exp_single.data

    visual = shap.plots.force(base_value, shap_values, features=features, show=False)
    shap.save_html(output_html, visual)

    print(f"Explicación SHAP guardada en: {output_html} (class_idx={class_idx})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="model_distil", help="Carpeta del modelo")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Texto a explicar")
    group.add_argument("--text_file", type=str, help="Ruta a un fichero de texto")
    parser.add_argument("--class_idx", type=int, default=None, help="Índice de la clase para force plot. Si no se indica, usa la clase predicha")
    parser.add_argument("--output_html", type=str, default="shap_text_explanation.html", help="Ruta de salida para el HTML interactivo")

    args = parser.parse_args()

    # Obtener el texto
    if args.text is not None:
        text = args.text
    else:
        # Leer desde fichero
        if not os.path.exists(args.text_file):
            raise FileNotFoundError(f"No se encuentra el fichero: {args.text_file}")
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()

    print("Texto a explicar:")
    print(text)
    print("-" * 80)

    # Cargar modelo, tokenizer y nombres de clases
    pipe, tokenizer, label_names = load_model_and_tokenizer(args.model_dir)

    # Calcular explicación SHAP
    print("Calculando explicación SHAP para el texto...")
    explanation = explain_text_with_shap(pipe, tokenizer, label_names, text)

    # Guardar gráfico interactivo en HTML
    save_explanation_html(explanation, args.output_html, args.class_idx)

    print("Listo.")


if __name__ == "__main__":
    main()
