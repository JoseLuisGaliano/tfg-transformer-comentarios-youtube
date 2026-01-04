import pandas as pd
import math
from collections import Counter
import matplotlib.pyplot as plt

# Calcula la entropía de Shannon de una lista de tokens
def shannon_entropy(tokens):
    if not tokens:
        return 0.0
    
    total = len(tokens)
    counts = Counter(tokens)
    ent = 0.0
    for token, cnt in counts.items():
        p = cnt / total
        ent -= p * math.log(p, 2)  # log base 2 -> bits
    return ent


def calculate_entropy_distribution(df):
    # Calcular entropía para cada fila
    df = df.copy();
    df['entropy'] = df['tokens'].apply(shannon_entropy)
    
    # Ordenar por entropía ascendente
    df = df.sort_values(by='entropy', ascending=True).reset_index(drop=True)
    
    df.to_csv('dataset_filter3_entropy.csv', index=False)
    
    # Estadísticas descriptivas
    entropy_stats = df['entropy'].describe()
    print(entropy_stats)
    
    return df['entropy']

def main():
    dataset_path = 'dataset_filter3.csv'
    df = pd.read_csv(dataset_path)
    
    # Calcular distribución de entropías
    entropies = calculate_entropy_distribution(df)
    
    # Calcular percentiles para ayudar a definir los umbrales
    min_entropy_percentile = entropies.quantile(0.25)  # Percentil 25
    max_entropy_percentile = entropies.quantile(0.75)  # Percentil 75
    
    print(f"Umbral mínimo de entropía (percentil 25): {min_entropy_percentile}")
    print(f"Umbral máximo de entropía (percentil 75): {max_entropy_percentile}")

if __name__ == "__main__":
    main()

