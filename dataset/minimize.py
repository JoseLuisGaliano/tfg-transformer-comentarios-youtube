import pandas as pd
import numpy as np

# Cargar dataset
df = pd.read_csv("dataset_filtered.csv")

# Número objetivo por clase
TARGET_PER_LABEL = 10000

# Columnas relevantes
LABEL_COL = "label"
VIDEO_ID_COL = "video_id"

# Iteramos por cada etiqueta para balancear
balanced_dfs = []

for label, group in df.groupby(LABEL_COL, sort=False):
    # Agrupamos por video_id, ya que siempre tienen el mismo label
    video_groups = list(group.groupby(VIDEO_ID_COL))
    
    # Calculamos tamaño por grupo (número de filas por video_id)
    group_sizes = np.array([len(g[1]) for g in video_groups])
    cum_sizes = np.cumsum(group_sizes)
    
    # Si el total excede el objetivo, cortamos grupos completos
    if len(group) > TARGET_PER_LABEL:
        # Encontramos cuántos grupos mantener para alcanzar o sobrepasar ligeramente el target
        n_keep = np.searchsorted(cum_sizes, TARGET_PER_LABEL)
        subset = pd.concat([g[1] for g in video_groups[:n_keep]])
    else:
        subset = group  # no alcanzamos el target, conservamos todas
    
    balanced_dfs.append(subset)

# Combinamos todo
df_balanced = pd.concat(balanced_dfs, ignore_index=True)

# Guardamos
df_balanced.to_csv("dataset.csv", index=False)

print("Dataset balanceado guardado como dataset.csv")
