import pandas as pd
import ast

# 1-gram diversity = (# de tokens únicos) / (# de tokens totales)
def unigram_diversity(tokens) -> float:

    if not tokens:
        return 0.0
    total = len(tokens)
    unique = len(set(tokens))
    return unique / total


def main():
    dataset_path = 'dataset_filter3_entropy.csv'
    df = pd.read_csv(dataset_path)

    # Convertir 'tokens' en una lista real
    df["tokens"] = df["tokens"].apply(ast.literal_eval)

    print("Calculando 1-gram diversity por comentario...")
    df["ngram"] = df["tokens"].apply(unigram_diversity)

    df.to_csv('dataset_filter3_ngram.csv', index=False)
    print("Listo")


if __name__ == "__main__":
    main()
