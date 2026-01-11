import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

CLASES = [
    "viajes",
    "musica",
    "ciencia",
    "cine",
    "deportes",
    "tecnologia",
    "noticias",
    "vehiculos",
    "animales",
]

print("Cargando embeddings...")
embeddings = np.load("embeddings.npy")

print("Cargando labels del dataset...")
df = pd.read_csv("./dataset.csv")

labels = df["label"].astype(str).tolist()

print("Reduciendo dimensionalidad con PCA...")
pca = PCA(n_components=2, random_state=0)
embeddings_2d = pca.fit_transform(embeddings)

# Mapeo etiqueta -> color
cmap = plt.get_cmap("tab10")
class_to_color = {cls: cmap(i) for i, cls in enumerate(CLASES)}

labels_arr = np.array(labels)

plt.figure(figsize=(10, 7))

for cls in CLASES:
    mask = labels_arr == cls
    if not mask.any():
        continue
    pts = embeddings_2d[mask]
    plt.scatter(
        pts[:, 0],
        pts[:, 1],
        alpha=0.6,
        s=18,
        color=class_to_color[cls],
        label=cls,
    )

plt.title("Visualización PCA 2D de embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, linewidth=0.5, alpha=0.4)
plt.legend(
    title="Etiqueta",
    fontsize=9,
    title_fontsize=10,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0.0,
)

plt.tight_layout(rect=[0, 0, 0.82, 1])
plt.savefig("visualizacion_pca_2d.png", dpi=200)
print("Gráfica guardada en visualizacion_pca_2d.png")
