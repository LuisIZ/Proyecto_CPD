"""
Proyecto CS4052 - Computacion Paralela y Distribuida (UTEC 2026-II)
Caso 5: KNN en paralelo con MPI (mpi4py) - Grupo 7

Version beta 2 (cambios respecto a la beta 1):
    - Distancias con NumPy: para cada punto de prueba se calculan de una vez
      las distancias a todo el bloque local (misma formula euclidiana).
    - La mezcla de candidatos y la votacion del maestro tambien con NumPy.
    - Si un bloque tiene menos de k puntos, los candidatos que faltan quedan con
      distancia +inf y no ganan nunca.
    - Medicion de tiempos por fase (bcast, scatter, computo, gather, maestro)
      con MPI.Wtime() y salida a un archivo CSV. Cada ejecucion mide UNA
      repeticion (--rep r): W rondas de calentamiento con los dos modos
      intercalados, que se descartan, y una corrida medida por modo (W por
      tiempo, ver knn_mpi_beta3.py).
      Modo A: sin barreras (tiempo de ejecucion T_ej).
      Modo B: con comm.Barrier() antes de cada marca (desglose por fase).
      El orden de los modos alterna con la repeticion (A,B si r es par; B,A si es impar).

Ejecucion:
    mpirun -np 4 python knn_mpi_beta2.py --rep 0 --csv resultados/tiempos.csv
    mpirun -np 4 python knn_mpi_beta2.py --validar
"""
from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import argparse
import os
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def knn_predict(test_point, X_train, y_train, k):
    # funcion del codigo base (solo se usa para validar)
    distances = [np.sqrt(np.sum((test_point - x) ** 2)) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    return Counter(k_labels).most_common(1)[0][0]


def k_vecinos_locales(X_test, X_q, y_q, idx_q, k):
    # pasos 3 y 4: para cada punto de prueba, distancias a todo mi bloque (NumPy).
    # Si el bloque tiene menos de k puntos, los candidatos que faltan quedan con
    # distancia +inf (etiqueta e indice -1) y no ganan nunca en la mezcla.
    n_te = len(X_test)
    m = min(k, len(X_q))
    dist = np.full((n_te, k), np.inf)
    etiq = np.full((n_te, k), -1, dtype=int)
    glob = np.full((n_te, k), -1, dtype=int)
    for i in range(n_te):
        D = np.sqrt(((X_q - X_test[i]) ** 2).sum(axis=1))   # paso 3: 3 FLOP por caracteristica
        I = np.argsort(D, kind='stable')[:m]                # paso 4: ordenar y tomar k
        dist[i, :m], etiq[i, :m], glob[i, :m] = D[I], y_q[I], idx_q[I]
    return dist, etiq, glob


def mezclar_y_votar(C, k):
    # paso 6: juntar los p*k candidatos de cada punto de prueba
    dist = np.hstack([c[0] for c in C])
    etiq = np.hstack([c[1] for c in C])
    glob = np.hstack([c[2] for c in C])
    orden = np.lexsort((glob, dist), axis=1)          # por distancia y luego indice global
    L = np.take_along_axis(etiq, orden, axis=1)[:, :k]
    # votacion: gana la etiqueta mas repetida; si hay empate, la del vecino mas cercano
    # (igual que Counter(...).most_common(1) del codigo base)
    cuenta = (L[:, :, None] == L[:, None, :]).sum(axis=2)
    return L[np.arange(len(L)), np.argmax(cuenta, axis=1)]


def una_corrida(X_train, y_train, X_test, y_test, k, fases):
    # una ejecucion completa del Algorithm 2 (distribuido); devuelve los tiempos (en el maestro)
    comm.Barrier()                                    # inicio alineado (fuera de la medicion)
    c0 = time.process_time()                          # CPU de este proceso (columna cpu)
    t0 = MPI.Wtime()
    X_te, k_l = comm.bcast((X_test, k), root=0)       # 1. Broadcast de la data de prueba
    if fases: comm.Barrier()
    t1 = MPI.Wtime()
    if rank == 0:                                     # 2. Scatter de la data de entrenamiento
        idx = np.arange(len(X_train))
        bloques = list(zip(np.array_split(X_train, size),
                           np.array_split(y_train, size),
                           np.array_split(idx, size)))
    else:
        bloques = None
    X_q, y_q, idx_q = comm.scatter(bloques, root=0)
    if fases: comm.Barrier()
    t2 = MPI.Wtime()
    C_q = k_vecinos_locales(X_te, X_q, y_q, idx_q, k_l)   # 3-4. k vecinos locales
    t3 = MPI.Wtime()
    if fases: comm.Barrier()
    t4 = MPI.Wtime()
    C = comm.gather(C_q, root=0)                      # 5. Gather de los candidatos
    t5 = MPI.Wtime()
    y_pred = acc = None
    if rank == 0:
        y_pred = mezclar_y_votar(C, k_l)              # 6. k globales y votacion
        acc = np.mean(y_pred == y_test)               # 7. Precision
    tf = MPI.Wtime()
    c1 = time.process_time()
    # fuera de la region medida: tiempo de computo de cada proceso y fraccion de CPU que tuvo
    # desde t0 hasta que termino su parte (con MPI los procesos que esperan siguen usando CPU)
    info = comm.gather((t3 - t2, (c1 - c0) / (tf - t0)), root=0)
    if rank == 0:
        t_comp = [a for a, _ in info]
        return dict(T_ej=tf - t0, T_bcast=t1 - t0, T_scatter=t2 - t1,
                    T_comp_max=max(t_comp), T_comp_prom=float(np.mean(t_comp)),
                    T_gather=t5 - t4, T_comp2=tf - t5, acc=acc, cpu=min(b for _, b in info)), y_pred
    return None, None


T_CAL, W_MAX = 0.25, 30                  # s de calentamiento por modo y tope de rondas


def calentamientos(d):
    # W por defecto: rondas de calentamiento (en cada una se corre una vez cada modo) para juntar
    # unos T_CAL s por modo, entre 1 y W_MAX; d = lo que tardo una corrida de la primera ronda
    return int(min(W_MAX, max(1, np.ceil(T_CAL / max(d, 1e-9)))))


def carga():
    # carga de la maquina (promedio del ultimo minuto, al empezar); os.getloadavg no existe en Windows.
    # El driver pasa en cambio --carga: los nucleos que usaban otros programas justo antes del mpirun
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return float("nan")


COLUMNAS = ["version", "n_tr", "n_te", "p", "rep", "modo", "W", "T_ej", "T_bcast", "T_scatter",
            "T_comp_max", "T_comp_prom", "T_gather", "T_comp2", "acc", "carga", "cpu", "hora"]

parser = argparse.ArgumentParser()
parser.add_argument("--rep", type=int, default=0)           # indice de la repeticion (lo pone el driver)
parser.add_argument("--calentamientos", type=int, default=None)   # rondas W (si no, por tiempo)
parser.add_argument("--carga", type=float, default=None)    # carga externa (la mide el driver)
parser.add_argument("--csv", default="")
parser.add_argument("--validar", action="store_true")
args = parser.parse_args()

# 0. Datos iniciales en el maestro (igual que el codigo base)
if rank == 0:
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )
    k = 3
    pares = len(X_train) * len(X_test)
    carga_ini = args.carga if args.carga is not None else carga()
else:
    X_train = y_train = X_test = y_test = k = pares = None
pares = comm.bcast(pares, root=0)        # todos hacen las mismas corridas (fuera de la medicion)

modos = [("A", False), ("B", True)]
if args.rep % 2 == 1:
    modos.reverse()                      # repeticiones impares: primero B y luego A


def ronda():
    # una corrida de cada modo, en el orden de la repeticion: [(modo, tiempos, y_pred)] en el maestro
    return [(modo, *una_corrida(X_train, y_train, X_test, y_test, k, fases)) for modo, fases in modos]


# W rondas de calentamiento con los modos intercalados (se descartan) y despues la ronda medida
W, hechas = args.calentamientos, 0
if W is None:
    W = 0
    if pares <= 2e8:                     # W por tiempo: el maestro cronometra la primera ronda
        t0 = MPI.Wtime()
        ronda()
        W = comm.bcast(calentamientos((MPI.Wtime() - t0) / len(modos)) if rank == 0 else None, root=0)
        hechas = 1
for _ in range(hechas, W):
    ronda()
medida = ronda()

if rank == 0:
    filas, preds = [], {}
    for modo, t, y_pred in medida:
        t.update(version=2, n_tr=len(X_train), n_te=len(X_test), p=size, rep=args.rep, modo=modo,
                 W=W, carga=carga_ini, hora=time.strftime("%Y-%m-%dT%H:%M:%S"))
        filas.append(t)
        preds[modo] = y_pred
    tiempos = "  ".join(f"T_ej({f['modo']}) = {f['T_ej']:.4f}" for f in filas)
    print(f"p = {size}  n_tr = {len(X_train)}  n_te = {len(X_test)}  Accuracy: {filas[0]['acc']:.4f}  "
          f"{tiempos} sec (beta 2, rep {args.rep}, W = {W}, cpu = {min(f['cpu'] for f in filas):.2f})")
    if args.csv:
        nuevo = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
        with open(args.csv, "a") as f:
            if nuevo:
                f.write(",".join(COLUMNAS) + "\n")
            for fila in filas:
                f.write(",".join(str(fila[c]) for c in COLUMNAS) + "\n")
    if args.validar:
        # validacion: mismas predicciones (de cada modo) que el codigo secuencial del profesor
        y_sec = np.array([knn_predict(x, X_train, y_train, k) for x in X_test])
        print(f"Predicciones iguales al codigo base: {all(np.array_equal(y, y_sec) for y in preds.values())}")
