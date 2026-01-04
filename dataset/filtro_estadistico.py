import pandas as pd

# UMBRALES
# Entropy
MIN_ENTROPY = 3.6
MAX_ENTROPY = 4.38
# N-gram Diversity
MIN_NGRAM = 0.4
MAX_NGRAM = 1
# Perplexity
MIN_PERPLEXITY = 10
MAX_PERPLEXITY = 1000


def main():
    
    df = pd.read_csv("dataset_filter3_ppl.csv")
    original_rows = len(df)

    df_filtrado = df[
        (df["entropy"]    >= MIN_ENTROPY)    & (df["entropy"]    <= MAX_ENTROPY) &
        (df["ngram"]      >= MIN_NGRAM)      & (df["ngram"]      <= MAX_NGRAM) &
        (df["perplexity"] >= MIN_PERPLEXITY) & (df["perplexity"] <= MAX_PERPLEXITY)
    ]

    filtered_rows = len(df_filtrado)

    df_filtrado.to_csv("dataset_filtered.csv", index=False)

    print(f"Filas originales: {original_rows}")
    print(f"Filas después del filtrado: {filtered_rows}")

if __name__ == "__main__":
    main()
