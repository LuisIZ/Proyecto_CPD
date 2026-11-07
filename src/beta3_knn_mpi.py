"""Beta 3: Beta 2 con el paso 6 optimizado (Algoritmo 2 del informe).

comm.reduce reemplaza a comm.gather: cada nivel del árbol fusiona dos listas y
solo reenvía k pares por punto de prueba. mpi4py usa un árbol por defecto para
reducir objetos (mpi4py.rc.fast_reduce).

Las fusiones ocurren dentro de comm.reduce. Para no contarlas como comunicación,
se miden con un cronómetro propio y se restan. La medición sigue las mismas
reglas de la Beta 2.

Uso: mpiexec -n 4 python beta3_knn_mpi.py [--n 1797] [--csv salida.csv]
"""
import argparse

import numpy as np
from mpi4py import MPI

from knn_comun import (K, N_BASE, armar_fila, cargar_datos, guardar_csv, repartir,
                       seleccionar_k, vecinos_locales, votar)


def corrida(comm, datos, X_tr, y_tr, y_te):
    """Ejecuta el Algoritmo 2 una vez y devuelve sus tiempos."""
    rank, p = comm.Get_rank(), comm.Get_size()
    bloques = y_pred = accuracy = None
    t_total = t_fusion = 0.0

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

    def fusionar(a, b):
        """Operador del informe: los k pares de menor distancia de a y b."""
        nonlocal t_fusion
        inicio = MPI.Wtime()
        unidos = [np.concatenate(partes, axis=1) for partes in zip(a, b)]
        resultado = seleccionar_k(*unidos, k)
        t_fusion += MPI.Wtime() - inicio
        return resultado

    t0 = MPI.Wtime()
    vecinos = comm.reduce(locales, op=fusionar, root=0)  # pasos 5 y 6
    t_recoleccion = MPI.Wtime() - t0 - t_fusion

    t_seleccion = t_fusion
    if rank == 0:
        t0 = MPI.Wtime()
        y_pred = votar(vecinos[1])
        accuracy = np.mean(y_pred == y_te)              # paso 7
        t_seleccion += MPI.Wtime() - t0
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
        fila = armar_fila("reduccion", args, comm.Get_size(), len(X_tr), medida, por_proceso)
        print(f"p={fila['p']} n={args.n} accuracy={medida['accuracy']:.4f} "
              f"t_total={medida['t_total']:.4f}s")
        if args.csv:
            guardar_csv(args.csv, fila)
        if args.pred:
            np.save(args.pred, medida["y_pred"])


if __name__ == "__main__":
    main()
