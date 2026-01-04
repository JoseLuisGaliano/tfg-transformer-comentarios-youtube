import pandas as pd
import re
import emoji
import langid
import ast

def filter_by_token_length(df, min_tokens=3, max_tokens=256):    
    # Calcular la longitud de cada lista de tokens
    df['tokens'] = df["tokens"].apply(ast.literal_eval)
    df['num_tokens'] = df['tokens'].apply(len)
    
    # Filtrar filas fuera de los límites
    df_filtered = df[(df["num_tokens"] >= min_tokens) & (df["num_tokens"] <= max_tokens)]

    # Ordenar por cantidad de tokens ascendente
    df_filtered = df_filtered.sort_values(by='num_tokens', ascending=True).reset_index(drop=True)

    return df_filtered


def filter_symbolic_noise(df, max_non_alphabetic_ratio=0.3, max_emoji_ratio=0.3):

    def is_symbolic_noise(tokens):     
        # Permitimos letras, números y caracteres como guiones y apóstrofes
        def is_alfabetico_o_valido(token):
            return bool(re.match(r'^[a-zA-Z0-9áéíóúÁÉÍÓÚñÑ\-_\'’]+$', token))
        
        # Contar la cantidad de caracteres no alfabéticos y emojis
        non_alphabetic_count = sum(1 for token in tokens if not is_alfabetico_o_valido(token))
        emoji_count = sum(1 for token in tokens if emoji.is_emoji(token))
        
        # Calcular la proporción de caracteres no alfabéticos y emojis
        non_alphabetic_ratio = non_alphabetic_count / len(tokens) if len(tokens) > 0 else 0
        emoji_ratio = emoji_count / len(tokens) if len(tokens) > 0 else 0
        
        # Si la proporción de caracteres no alfabéticos o emojis es mayor que los umbrales, es ruido   
        if non_alphabetic_ratio > max_non_alphabetic_ratio or emoji_ratio > max_emoji_ratio:
            return True
                        
        return False
    
    # Filtrar comentarios que sean considerados "ruido simbólico"
    df_filtered = df[~df['tokens'].apply(is_symbolic_noise)]  
    return df_filtered


def filter_by_language(df, language='es'):

    def is_spanish(comment):
        # Detectar el idioma del comentario utilizando langid
        lang, _ = langid.classify(comment)  # 'langid.classify()' devuelve un par (idioma, probabilidad)        
        # Comparar el idioma detectado con el código de idioma 'es' para español
        return lang == language  
    
    # Filtrar los comentarios que están en español
    df_filtered = df[df['text'].apply(is_spanish)]
    
    return df_filtered


def main():
    # Cargar el dataset
    dataset_path = 'dataset_tokenizado.csv'
    df = pd.read_csv(dataset_path)
    
    # Filtrar el dataset según la longitud de los tokens
    print("Filtrando por longitud...")
    df_filter1 = filter_by_token_length(df)
    
    # Filtrar el dataset por ruido simbólico
    print("Filtrando por ruido simbólico...")
    df_filter2 = filter_symbolic_noise(df_filter1)
    
    # Filtrar el dataset por idioma (sólo español)
    print("Filtrando por idioma...")
    df_filter3 = filter_by_language(df_filter2)
    
    # Guardar el dataset filtrado (por pasos para poder hacer analisis)
    df_filter1.to_csv('dataset_filter1.csv', index=False)
    df_filter2.to_csv('dataset_filter2.csv', index=False)
    df_filter3.to_csv('dataset_filter3.csv', index=False)


if __name__ == "__main__":
    main()
