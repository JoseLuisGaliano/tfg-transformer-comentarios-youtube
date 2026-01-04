import pandas as pd

# Cargar el CSV
df = pd.read_csv('predicciones_test_transformer.csv')

# Filtrar solo los errores (donde pred != label)
df_errores = df[df['pred'] != df['label']]

# Guardar el resultado
df_errores.to_csv('errores_clasificacion.csv', index=False)

print(f"Total de filas originales: {len(df)}")
print(f"Filas con errores: {len(df_errores)}")
print(f"Precisión del modelo: {((len(df) - len(df_errores)) / len(df) * 100):.2f}%")

