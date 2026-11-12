"""
Proyecto CS4052 - Caso 5 KNN - Grupo 7
Ping-pong entre dos procesos (aporte de Luis): mide la latencia alfa y el tiempo por
dato beta del modelo de comunicacion T(m) = alfa + m*beta, con m datos float64.
    - Tamanos: m de 1 a 10^6 datos, en escala logaritmica.
    - Dos variantes:
        bufer:   comm.Send / comm.Recv con arreglos de NumPy (sin serializar)
        objetos: comm.send / comm.recv con pickle, que es lo que usan las colectivas
                 en minuscula de las betas (bcast, scatter y gather)
    - Para cada tamano y variante se miden REPS repeticiones; cada una cronometra varias
      idas y vueltas seguidas y se queda con la mitad del promedio (el tiempo de ida).
      Se reporta la mediana de las repeticiones (y el minimo y el maximo).
    - El ajuste de alfa y beta por minimos cuadrados lo hace analisis.py.

Ejecucion:  mpirun -np 2 python ping_pong.py --csv resultados/pingpong.csv
"""
from mpi4py import MPI
import numpy as np
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

TAMANOS = np.unique(np.logspace(0, 6, 13).astype(int))   # 1, 3, 10, 31, ..., 10^6 datos float64
REPS = 7                 # repeticiones por tamano y variante (se reporta la mediana)
CALENTAMIENTO = 5        # idas y vueltas que se descartan antes de medir cada tamano


def ida(mensaje, variante, viajes):
    # tiempo de ida de un mensaje: la mitad de una ida y vuelta, promediada en 'viajes' seguidas
    recibido = np.empty_like(mensaje)
    comm.Barrier()
    t0 = MPI.Wtime()
    for _ in range(viajes):
        if variante == "bufer":
            if rank == 0:
                comm.Send(mensaje, dest=1)
                comm.Recv(recibido, source=1)
            else:
                comm.Recv(recibido, source=0)
                comm.Send(recibido, dest=0)
        else:
            if rank == 0:
                comm.send(mensaje, dest=1)
                recibido = comm.recv(source=1)
            else:
                recibido = comm.recv(source=0)
                comm.send(recibido, dest=0)
    return (MPI.Wtime() - t0) / (2 * viajes)


parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="")
args = parser.parse_args()
if comm.Get_size() != 2:
    raise SystemExit("ping_pong.py necesita exactamente 2 procesos: mpirun -np 2 python ping_pong.py")

filas = []
for m in TAMANOS:
    mensaje = np.ones(m)
    viajes = int(max(10, min(1000, 10 ** 6 // m)))        # mas viajes con mensajes chicos
    for variante in ("bufer", "objetos"):
        ida(mensaje, variante, CALENTAMIENTO)              # calentamiento (se descarta)
        t = [ida(mensaje, variante, viajes) for _ in range(REPS)]
        filas.append((variante, int(m), 8 * int(m), float(np.median(t)), min(t), max(t), REPS, viajes))

if rank == 0:
    print(f"{'variante':>8} {'m (datos)':>10} {'bytes':>9} {'ida (us)':>10}")
    for variante, m, b, t, *_ in filas:
        print(f"{variante:>8} {m:>10} {b:>9} {t * 1e6:>10.2f}")
    if args.csv:
        with open(args.csv, "w") as f:
            f.write("variante,m,bytes,t_ida,t_min,t_max,reps,viajes\n")
            for fila in filas:
                f.write(",".join(str(x) for x in fila) + "\n")
