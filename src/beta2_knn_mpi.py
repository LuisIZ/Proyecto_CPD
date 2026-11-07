"""Beta 2: Beta 1 con medición de tiempos y de FLOP.

No se usan barreras, porque agregan overhead a lo que se mide. Por eso:
- El algoritmo se ejecuta dos veces y solo se mide la segunda. Sin esa corrida
  de calentamiento, el maestro empieza a contar mientras otros procesos siguen
  importando librerías, y esa espera aparece como comunicación.
- t_total y t_comm salen del reloj del maestro, que es el que ve el tiempo
  completo del algoritmo. t_comm_max guarda además el máximo entre procesos, que
  incluye la espera de los otros procesos por el maestro.
- t_comp_paralelo es el máximo entre procesos, porque el cómputo no incluye
  esperas y el proceso más lento es el que determina el tiempo.
- t_recoleccion incluye la espera del maestro por el proceso más lento.

Uso: mpiexec -n 4 python beta2_knn_mpi.py [--n 1797] [--csv salida.csv]
"""
import argparse

import numpy as np
from mpi4py import MPI

from knn_comun import (K, N_BASE, armar_fila, cargar_datos, combinar, guardar_csv,
                       repartir, vecinos_locales, votar)


def corrida(comm, datos, X_tr, y_tr, y_te):
    """Ejecuta el Algoritmo 1 una vez y devuelve sus tiempos."""
    rank, p = comm.Get_rank(), comm.Get_size()
    bloques = y_pred = accuracy = None
    t_seleccion = t_total = 0.0

    t_inicio = MPI.Wtime()

    X_te, k = comm.bcast(datos, root=0)                 # paso 1
    t_bcast = MPI.Wtime() - t_inicio

    t0 = MPI.Wtime()
    if rank == 0:
        bloques = repartir(X_tr, y_tr, p)
    X_b, y_b, i_b = comm.scatter(bloques, root=0)       # paso 2
    t_scatter = MPI.Wtime() - t0

    t0 = MPI.Wtime()
    locales, t_dist, t_sort = vecinos_locales(X_te, X_b, y_b, i_b, k, MPI.Wtime)  # pasos 3 y 4
    t_comp_paralelo = MPI.Wtime() - t0

    t0 = MPI.Wtime()
    candidatos = comm.gather(locales, root=0)           # paso 5
    t_recoleccion = MPI.Wtime() - t0

    if rank == 0:
        t0 = MPI.Wtime()
        _, etiquetas, _ = combinar(candidatos, k)       # paso 6
        y_pred = votar(etiquetas)
        accuracy = np.mean(y_pred == y_te)              # paso 7
        t_seleccion = MPI.Wtime() - t0
        t_total = MPI.Wtime() - t_inicio

    return {
        "t_total": t_total, "t_seleccion": t_seleccion, "t_bcast": t_bcast,
        "t_scatter": t_scatter, "t_recoleccion": t_recoleccion, "accuracy": accuracy,
        "y_pred": y_pred, "n_te": len(X_te),
        "por_proceso": (t_comp_paralelo, t_dist, t_sort,
                        3 * X_b.shape[1] * len(X_b) * len(X_te),
                        t_bcast + t_scatter + t_recoleccion),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_BASE)
    parser.add_argument("--repeticion", type=int, default=0)
    parser.add_argument("--csv", help="archivo donde agregar la medición")
    parser.add_argument("--pred", help="archivo .npy para guardar las predicciones")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    datos = X_tr = y_tr = y_te = None
    if rank == 0:
        X_tr, X_te, y_tr, y_te = cargar_datos(args.n)
        datos = (X_te, K)

    corrida(comm, datos, X_tr, y_tr, y_te)              # calentamiento, sin medir
    medida = corrida(comm, datos, X_tr, y_tr, y_te)

    por_proceso = comm.gather(medida["por_proceso"], root=0)

    if rank == 0:
        fila = armar_fila("gather", args, comm.Get_size(), len(X_tr), medida, por_proceso)
        print(f"p={fila['p']} n={args.n} accuracy={medida['accuracy']:.4f} "
              f"t_total={medida['t_total']:.4f}s")
        if args.csv:
            guardar_csv(args.csv, fila)
        if args.pred:
            np.save(args.pred, medida["y_pred"])


if __name__ == "__main__":
    main()
