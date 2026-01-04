import pandas as pd
import re
from unidecode import unidecode

def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = unidecode(texto)
    texto = texto.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def es_solo_emojis(texto):
    emoji_pattern = re.compile(r'^[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+$', flags=re.UNICODE)
    return bool(emoji_pattern.fullmatch(texto))

def es_solo_enlaces(texto):
    enlace_pattern = re.compile(r'^(https?://[^\s]+|www\.[^\s]+)$', flags=re.IGNORECASE)
    return bool(enlace_pattern.fullmatch(texto.strip()))

def limpiar_author_name(nombre):
    # Quita el carácter '@' si está al principio
    return str(nombre)[1:] if str(nombre).startswith('@') else str(nombre)

def limpiar_topic_categories(categorias):
    # Extrae el nombre de la página de Wikipedia
    if pd.isna(categorias):
        return categorias
    # Separar por '|', limpiar cada parte
    partes = [re.sub(r'^https?://(?:[^/]+/)+wiki/', '', cat.strip()) for cat in str(categorias).split('|')]
    return ' | '.join(partes)

df = pd.read_csv('./datasetRAW.csv')
df = df.drop_duplicates()
df['text'] = df['text'].apply(limpiar_texto)

condicion = (
    (df['text'].str.strip() == '') |
    (df['text'].apply(es_solo_emojis)) |
    (df['text'].apply(es_solo_enlaces))
)
df = df[~condicion]

df['author_name'] = df['author_name'].apply(limpiar_author_name)

if 'topicCategories' in df.columns:
    df['topicCategories'] = df['topicCategories'].apply(limpiar_topic_categories)

df.to_csv('./dataset_limpio.csv', index=False)
