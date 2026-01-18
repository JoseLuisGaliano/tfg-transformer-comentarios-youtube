import argparse
import pandas as pd


def main():

    df = pd.read_csv("dataset.csv")

    # Columnas mínimas necesarias
    required = [
        "text",
        "author_id",
        "channel_id",
        "num_tokens",
        "entropy",
        "ngram",
        "perplexity",
        "label",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {missing}")

    df["text"] = df["text"].astype(str)
    df["text_len_chars"] = df["text"].str.len()

    numeric_cols = ["num_tokens", "ngram", "entropy", "perplexity"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Agregación por label
    out = (
        df.groupby("label", dropna=False)
        .agg(
            longitud_media_caracteres=("text_len_chars", "mean"),
            longitud_media_tokens=("num_tokens", "mean"),
            diversidad_unigramas_media=("ngram", "mean"),
            entropia_media=("entropy", "mean"),
            perplexity_media=("perplexity", "mean"),
            canales_unicos=("channel_id", pd.Series.nunique),
            autores_unicos=("author_id", pd.Series.nunique),
            n_filas=("label", "size"),
        )
        .reset_index()
    )

    # Redondeo razonable
    float_cols = [
        "longitud_media_caracteres",
        "longitud_media_tokens",
        "diversidad_unigramas_media",
        "entropia_media",
        "perplexity_media",
    ]
    out[float_cols] = out[float_cols].round(4)

    out.to_csv("stats_por_label.csv", index=False, encoding="utf-8")
    print(f"OK: guardado stats_por_label.csv ({len(out)} labels)")


if __name__ == "__main__":
    main()

