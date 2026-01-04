import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Cargar el dataset para acceder a los tokens
df = pd.read_csv('./dataset_procesado.csv')

# Generar embeddings con SentenceTransformer
print("Cargando modelo de SentenceTransformer...")
modelo = SentenceTransformer('distiluse-base-multilingual-cased-v2')

print("Generando embeddings...")
frases = df["tokens"].tolist()
embeddings = modelo.encode(frases, show_progress_bar=True)

# Guardar los embeddings en un archivo numpy
np.save("embeddings.npy", embeddings)
print("Embeddings guardados!")
