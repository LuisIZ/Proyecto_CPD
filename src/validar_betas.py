"""Compara las predicciones de cada beta con las del código base.

Uso: python validar_betas.py [--p 1 2 3 4 7 8 16]
"""
import argparse
import os
import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np

from lanzador import RAIZ, ejecutar, ejecutar_sin_mpi

BETAS = ["beta1_knn_mpi.py", "beta2_knn_mpi.py", "beta3_knn_mpi.py"]
PRECISION_BASE = 0.9833


def predicciones_base():
    """Ejecuta el código base sin abrir su ventana de gráficos."""
    os.environ["MPLBACKEND"] = "Agg"
    resultado = runpy.run_path(str(RAIZ / "drive" / "knn_digits_sec.py"))
    return np.array(resultado["y_pred"]), float(resultado["accuracy"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, nargs="+", default=[1, 2, 3, 4, 7, 8, 16])
    args = parser.parse_args()

    y_base, precision = predicciones_base()
    print(f"Código base: accuracy={precision:.4f}\n")

    fallas = 0
    with tempfile.TemporaryDirectory() as carpeta:
        ruta = Path(carpeta) / "pred.npy"
        print(f"{'versión':<20}{'p':>4}{'accuracy':>10}{'distintas':>11}")
        ejecutar_sin_mpi("secuencial_vectorizado.py", "--pred", ruta)
        fallas += comparar("secuencial", 1, np.load(ruta), y_base)
        for beta in BETAS:
            for p in args.p:
                ejecutar(beta, p, "--pred", ruta)
                fallas += comparar(beta[:5], p, np.load(ruta), y_base)

    print("\nTodas las versiones coinciden con el código base." if not fallas
          else f"\n{fallas} combinaciones no coinciden.")
    sys.exit(1 if fallas else 0)


def comparar(nombre, p, y_pred, y_base):
    """Imprime la comparación y devuelve 1 si hay diferencias."""
    precision = float(np.mean(y_pred == cargar_y_te()))
    distintas = int(np.sum(y_pred != y_base))
    correcto = distintas == 0 and round(precision, 4) == PRECISION_BASE
    marca = "" if correcto else "  <-- REVISAR"
    print(f"{nombre:<20}{p:>4}{precision:>10.4f}{distintas:>11}{marca}")
    return 0 if correcto else 1


def cargar_y_te():
    from knn_comun import cargar_datos
    return cargar_datos()[3]


if __name__ == "__main__":
    main()
