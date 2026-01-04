import argparse
import os
import sys

import pandas as pd
from ydata_profiling import ProfileReport


def main():
    ap = argparse.ArgumentParser(description="Genera un reporte EDA interactivo desde un CSV")
    ap.add_argument("--csv", required=True, help="Ruta al CSV de entrada")
    ap.add_argument("--out", default="reporte_datos.html", help="Ruta de salida del HTML")
    ap.add_argument("--title", default="Informe exploratorio de datos", help="Título del reporte")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print(f"No existe el archivo: {args.csv}", file=sys.stderr)
        sys.exit(1)

    print("Leyendo CSV...")
    df = pd.read_csv(args.csv)
    nrows, ncols = df.shape
    print(f"DataFrame cargado: {nrows:,} filas x {ncols:,} columnas")
    print(df.head)

    print("Generando reporte interactivo...")
    profile = ProfileReport(
        df,
        title=args.title,
        explorative=True,                  # activa paneles extra de exploración
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": False},
            "phi_k": {"calculate": True},
        },
        interactions={"continuous": True},   # matriz de interacciones
    )

    outpath = os.path.abspath(args.out)
    profile.to_file(outpath)
    print(f"Reporte creado: {outpath}")

if __name__ == "__main__":
    main()

