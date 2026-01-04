#!/usr/bin/env python
import argparse
import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bertviz import head_view, model_view


def load_text(text_path: str) -> str:
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="../../training/transformer-hfV4mini/outputs/model_distil", help="Carpeta donde está guardado el modelo")
    parser.add_argument("--text_path", type=str, default="ejemplo.txt", help="Ruta al fichero .txt con el texto a analizar")
    parser.add_argument("--output_dir", type=str, default="bertviz_outputs", help="Carpeta donde se guardarán los HTML")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Cargar texto
    text = load_text(args.text_path)
    if not text:
        raise ValueError(f"El fichero {args.text_path} está vacío o solo contiene espacios.")

    # 2. Cargar tokenizer y modelo
    print(f"Cargando modelo desde {args.model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    # 3. Tokenizar
    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    # 4. Ejecutar el modelo con output_attentions=True
    print("Ejecutando el modelo y recuperando las atenciones...")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions  # tuple: (num_layers, batch, num_heads, seq_len, seq_len)

    # 5. Convertir IDs a tokens (para visualización)
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # 6. Generar y guardar HEAD VIEW
    print("Generando head view de BERTViz...")
    head_html = head_view(
        attentions,
        tokens,
        html_action="return",  # importante: devolvemos el HTML como string
    )
    head_output_path = os.path.join(args.output_dir, "head_view.html")
    with open(head_output_path, "w", encoding="utf-8") as f:
        f.write(head_html.data)
    print(f"Head view guardado en: {head_output_path}")

    # 7. Generar y guardar MODEL VIEW
    print("Generando model view de BERTViz...")
    model_html = model_view(
        attentions,
        tokens,
        html_action="return",  # igual: devolvemos HTML como string
    )
    model_output_path = os.path.join(args.output_dir, "model_view.html")
    with open(model_output_path, "w", encoding="utf-8") as f:
        f.write(model_html.data)
    print(f"Model view guardado en: {model_output_path}")

    print("Listo.")


if __name__ == "__main__":
    main()

