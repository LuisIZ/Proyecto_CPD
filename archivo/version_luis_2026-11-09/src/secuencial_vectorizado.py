"""T_s: la misma rutina de distancias de las betas, en un proceso y sin MPI.

El código base no está vectorizado. Compararlo con las betas mezclaría el efecto
de la vectorización con el del paralelismo, por eso el speedup usa este tiempo.
El tiempo del código base se reporta aparte.

Uso: python secuencial_vectorizado.py [--n 1797] [--csv salida.csv]
"""
import argparse
import csv
import os
import time

import numpy as np

from knn_comun import K, N_BASE, cargar_datos, vecinos_locales, votar


def corrida(X_tr, X_te, y_tr, y_te):
    """Clasifica todos los puntos de prueba y devuelve el tiempo usado."""
    inicio = time.perf_counter()
    (_, etiquetas, _), _, _ = vecinos_locales(X_te, X_tr, y_tr, np.arange(len(X_tr)), K)
    y_pred = votar(etiquetas)
    accuracy = np.mean(y_pred == y_te)
    return time.perf_counter() - inicio, y_pred, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_BASE)
    parser.add_argument("--repeticion", type=int, default=0)
    parser.add_argument("--csv", help="archivo donde agregar la medición")
    parser.add_argument("--pred", help="archivo .npy para guardar las predicciones")
    args = parser.parse_args()

    datos = cargar_datos(args.n)
    corrida(*datos)                                     # calentamiento, igual que las betas
    t_total, y_pred, accuracy = corrida(*datos)

    print(f"n={args.n} accuracy={accuracy:.4f} t_total={t_total:.4f}s")
    if args.csv:
        nuevo = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="", encoding="utf-8") as archivo:
            escritor = csv.writer(archivo)
            if nuevo:
                escritor.writerow(["n", "repeticion", "t_total", "accuracy"])
            escritor.writerow([args.n, args.repeticion, t_total, accuracy])
    if args.pred:
        np.save(args.pred, y_pred)


if __name__ == "__main__":
    main()
