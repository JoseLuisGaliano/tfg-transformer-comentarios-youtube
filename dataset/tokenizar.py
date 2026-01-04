import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

df = pd.read_csv('./dataset_limpio.csv')
df = df.drop_duplicates()

# TOKENIZACIÓN
df['tokens'] = df['text'].apply(word_tokenize)

# ELIMINACIÓN DE STOPWORDS
stop_words = set(stopwords.words('spanish'))
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

df.to_csv('./dataset_tokenizado.csv', index=False)
