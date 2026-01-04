import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print("Cargando embeddings...")
embeddings = np.load("embeddings.npy")

print("Reduciendo dimensionalidad con PCA...")
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 7))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
plt.title("Visualización PCA 2D de embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.savefig("visualizacion_pca_2d.png")
