"""Mide la latencia a y el tiempo por dato b con un ping-pong entre 2 procesos.

Usa comm.send y comm.recv en minúscula, igual que las betas, así que el resultado
incluye el costo de serializar con pickle. El tamaño es la cantidad de números
float64 del mensaje. Las primeras iteraciones no se cuentan.

Uso: mpiexec -n 2 python ping_pong.py [--csv pingpong.csv]
"""
import argparse
import csv

import numpy as np
from mpi4py import MPI

TAMANOS = [2**i for i in range(0, 21, 2)]
CALENTAMIENTO = 20
REPETICIONES = 5


def medir(comm, mensaje):
    """Tiempo de ida de un mensaje: la mitad de un viaje de ida y vuelta."""
    rank = comm.Get_rank()
    viajes = max(20, min(500, 2_000_000 // mensaje.size))
    for i in range(CALENTAMIENTO + viajes):
        if i == CALENTAMIENTO:
            inicio = MPI.Wtime()
        if rank == 0:
            comm.send(mensaje, dest=1)
            comm.recv(source=1)
        else:
            comm.recv(source=0)
            comm.send(mensaje, dest=0)
    return (MPI.Wtime() - inicio) / (2 * viajes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="archivo donde guardar las mediciones")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    filas = []
    for tamano in TAMANOS:
        mensaje = np.ones(tamano)
        tiempos = [medir(comm, mensaje) for _ in range(REPETICIONES)]
        filas.append((tamano, float(np.median(tiempos))))

    if comm.Get_rank() == 0:
        for tamano, tiempo in filas:
            print(f"{tamano:>9} datos: {tiempo * 1e6:12.2f} us")
        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as archivo:
                escritor = csv.writer(archivo)
                escritor.writerow(["tamano", "t_mensaje"])
                escritor.writerows(filas)


if __name__ == "__main__":
    main()
