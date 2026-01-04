import pandas as pd

# Cargar los CSVs de errores
errores_a = pd.read_csv('errores_baseline.csv')
errores_b = pd.read_csv('errores_transformer.csv')

# Identificar filas únicas usando la columna 'text' como clave
textos_a = set(errores_a['text'])
textos_b = set(errores_b['text'])

# Calcular los conjuntos
solo_a = textos_a - textos_b
solo_b = textos_b - textos_a
mutuos = textos_a & textos_b

# Filtrar DataFrames
errores_solo_a = errores_a[errores_a['text'].isin(solo_a)]
errores_solo_b = errores_b[errores_b['text'].isin(solo_b)]
errores_mutuos_a = errores_a[errores_a['text'].isin(mutuos)]

# Guardar resultados
errores_solo_a.to_csv('errores_solo_baseline.csv', index=False)
errores_solo_b.to_csv('errores_solo_transformer.csv', index=False)
errores_mutuos_a.to_csv('errores_mutuos.csv', index=False)

# Estadísticas
print(f"Total errores modelo baseline: {len(errores_a)}")
print(f"Total errores modelo transformer: {len(errores_b)}")
print(f"\nErrores solo en baseline: {len(errores_solo_a)}")
print(f"Errores solo en transformer: {len(errores_solo_b)}")
print(f"Errores mutuos: {len(errores_mutuos_a)}")
print(f"\nPorcentaje de errores únicos baseline: {(len(errores_solo_a)/len(errores_a)*100):.2f}%")
print(f"Porcentaje de errores únicos transformer: {(len(errores_solo_b)/len(errores_b)*100):.2f}%")
print(f"Porcentaje de errores compartidos: {(len(errores_mutuos_a)/len(errores_a)*100):.2f}%")

