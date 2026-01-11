import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from io import StringIO
import json

LABELS = [
    "cine",
    "musica",
    "vehiculos",
    "ciencia",
    "tecnologia",
    "animales",
    "noticias",
    "deportes",
    "viajes",
]

def main():
    # 1. Cargar datasets ya divididos
    TRAIN_PATH = "train.csv"
    VAL_PATH = "val.csv"
    TEST_PATH = "test.csv"

    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    # Nos quedamos con las columnas relevantes
    X_train = df_train["text"].astype(str).tolist()
    y_train = df_train["label"].astype(str).tolist()

    X_val = df_val["text"].astype(str).tolist()
    y_val = df_val["label"].astype(str).tolist()

    X_test = df_test["text"].astype(str).tolist()
    y_test = df_test["label"].astype(str).tolist()

    # 2. Definir el pipeline TF-IDF + RegLog
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            multi_class="multinomial",
            n_jobs=-1
        ))
    ])

    # 3. Entrenar
    model.fit(X_train, y_train)

    # 4. Evaluar
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")


    buffer = StringIO()
    sys.stdout = buffer  # redirigir print a buffer temporal
    
    print("==== BASELINE TF-IDF + LogisticRegression ====")
    print(f"Accuracy:   {acc:.5f}")
    print(f"F1 (macro): {f1_macro:.5f}")
    print()

    print("---- Clasification report ----")
    print(classification_report(y_test, y_pred, digits=5))

    # 5. Matriz de confusión
    classes_sorted = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, y_pred, labels=classes_sorted)
    cm_df = pd.DataFrame(cm, index=sorted(list(set(y_test))), columns=sorted(list(set(y_test))))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de confusión - Baseline TF-IDF + RegLog")
    plt.tight_layout()
    plt.savefig("confusion_matrix_regression.png", dpi=300)
    plt.close()

    # 6. Top features por clase
    clf = model.named_steps["clf"]
    vectorizer = model.named_steps["tfidf"]
    feature_names = vectorizer.get_feature_names_out()
    classes = clf.classes_

    top_k = 10
    for i, cls in enumerate(classes):
        coefs = clf.coef_[i]
        top_idx = coefs.argsort()[::-1][:top_k]
        print(f"\n>>> TOP {top_k} rasgos para clase '{cls}':")
        for j in top_idx:
            print(f"{feature_names[j]:<30} peso={coefs[j]:.4f}")
            
    # 7. Guardar salida textual
    sys.stdout = sys.__stdout__  # restaurar stdout
    results_text = buffer.getvalue()
    buffer.close()

    with open("results_regression.txt", "w", encoding="utf-8") as f:
        f.write(results_text)

    print("Resultados guardados en 'results_regression.txt'")
    print("Matriz de confusión guardada en 'confusion_matrix_regression.png'")
    
    # 8. Predicciones + probabilidades a CSV para analisis
    proba = model.predict_proba(X_test)  # shape: (n_samples, n_classes)
    classes_order = list(classes)        # orden exacto de columnas en 'proba'

    # Guardar el orden de clases (para interpretar 'probabilidades')
    with open("orden_clases.json", "w", encoding="utf-8") as f:
        json.dump(classes_order, f, ensure_ascii=False, indent=2)
    print(f"Orden de clases guardado en 'orden_clases.json'")

    # Formatear a dos decimales en el CSV (representación corta)
    prob_strings = [
        json.dumps([float(f"{p:.2f}") for p in row])
        for row in proba
    ]

    # Construir DataFrame de salida
    out_df = pd.DataFrame({
        "text": df_test["text"].astype(str),
        "probabilidades": prob_strings,
        "pred": y_pred,
        "label": y_test
    })

    # Agrupar por label real
    out_df["label"] = pd.Categorical(out_df["label"], categories=LABELS)
    out_df = out_df.sort_values("label").reset_index(drop=True)

    out_df.to_csv("predicciones_test_regression.csv", index=False)
    print(f"CSV de predicciones guardado en 'predicciones_test_regression.csv'")

if __name__ == "__main__":
    main()

