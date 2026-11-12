"""Beta 1: KNN en paralelo con bcast, scatter y gather (Algoritmo 1 del informe).

Uso: mpiexec -n 4 python beta1_knn_mpi.py [--n 1797] [--pred salida.npy]
"""
import argparse

import numpy as np
from mpi4py import MPI

from knn_comun import K, N_BASE, cargar_datos, combinar, repartir, vecinos_locales, votar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_BASE)
    parser.add_argument("--pred", help="archivo .npy para guardar las predicciones")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank, p = comm.Get_rank(), comm.Get_size()

    datos = bloques = y_te = None
    if rank == 0:
        X_tr, X_te, y_tr, y_te = cargar_datos(args.n)
        datos = (X_te, K)
        bloques = repartir(X_tr, y_tr, p)

    X_te, k = comm.bcast(datos, root=0)             # paso 1
    X_b, y_b, i_b = comm.scatter(bloques, root=0)   # paso 2

    locales, _, _ = vecinos_locales(X_te, X_b, y_b, i_b, k)  # pasos 3 y 4

    candidatos = comm.gather(locales, root=0)       # paso 5

    if rank == 0:
        _, etiquetas, _ = combinar(candidatos, k)   # paso 6
        y_pred = votar(etiquetas)
        accuracy = np.mean(y_pred == y_te)          # paso 7
        print(f"p={p} n={args.n} accuracy={accuracy:.4f}")
        if args.pred:
            np.save(args.pred, y_pred)


if __name__ == "__main__":
    main()
