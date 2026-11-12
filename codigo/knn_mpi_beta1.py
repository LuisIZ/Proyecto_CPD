"""
Proyecto CS4052 - Computacion Paralela y Distribuida (UTEC 2026-II)
Caso 5: KNN en paralelo con MPI (mpi4py) - Grupo 7

Version beta 1: traduccion directa del algoritmo distribuido (Algorithm 2 del informe).
    - Sigue el DAG del enunciado: bcast de la data de prueba, scatter de la
      data de entrenamiento, k vecinos locales en cada proceso, gather de los
      candidatos y votacion por mayoria en el maestro (proceso 0).
    - El calculo de distancias es igual al del codigo base (knn_digits_sec.py):
      un bucle de Python con euclidean_distance por cada par.
    - Se mide con el mismo protocolo de las betas 2 y 3: cada ejecucion mide UNA
      repeticion (--rep r), con W rondas de calentamiento (los dos modos intercalados)
      y una corrida medida por modo (A: sin barreras, de ahi sale T_ej; B: con
      barreras para separar las fases). W por tiempo, ver knn_mpi_beta3.py.

Ejecucion:
    mpirun -np 4 python knn_mpi_beta1.py
    mpirun -np 4 python knn_mpi_beta1.py --validar   (compara con el secuencial)
    mpirun -np 4 python knn_mpi_beta1.py --rep 0 --csv resultados/tiempos.csv
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


def euclidean_distance(a, b):
    # misma funcion del codigo base: 3 FLOP por caracteristica (resta, producto, suma)
    return np.sqrt(np.sum((a - b) ** 2))


def knn_predict(test_point, X_train, y_train, k):
    # funcion del codigo base (solo se usa para validar)
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]


def k_vecinos_locales(X_test, X_q, y_q, idx_q, k):
    # pasos 3 y 4 del Algorithm 2: distancias de cada punto de prueba a mi bloque
    # y mis k vecinos mas cercanos (distancia, etiqueta, indice global).
    # Si el bloque tiene menos de k puntos, la lista queda mas corta: no hace
    # falta relleno, porque la mezcla junta listas de cualquier largo.
    C_q = []
    for x in X_test:
        distances = [euclidean_distance(x, t) for t in X_q]           # paso 3
        k_indices = np.argsort(distances, kind='stable')[:k]           # paso 4
        C_q.append([(distances[j], y_q[j], idx_q[j]) for j in k_indices])
    return C_q


def mezclar_y_votar(C, k):
    # paso 6: por cada punto de prueba, juntamos los p*k candidatos,
    # nos quedamos con los k de menor distancia y votamos
    n_te = len(C[0])
    y_pred = []
    for i in range(n_te):
        candidatos = [c for C_q in C for c in C_q[i]]
        candidatos.sort(key=lambda c: (c[0], c[2]))   # distancia y luego indice global
        k_labels = [c[1] for c in candidatos[:k]]
        y_pred.append(Counter(k_labels).most_common(1)[0][0])
    return np.array(y_pred)


def una_corrida(X_train, y_train, X_test, y_test, k, fases):
    # una ejecucion completa del Algorithm 2 (pasos 1 a 7); devuelve los tiempos (en el maestro)
    comm.Barrier()                       # todos empiezan juntos (fuera de la medicion)
    c0 = time.process_time()             # CPU de este proceso (columna cpu)
    t0 = MPI.Wtime()                     # tiempo inicial (en el maestro)

    # 1. Broadcast de la data de prueba y de k desde el maestro
    X_te, k_l = comm.bcast((X_test, k), root=0)
    if fases: comm.Barrier()
    t1 = MPI.Wtime()

    # 2. Scatter de la data de entrenamiento (n_tr puede no ser multiplo de p)
    if rank == 0:
        idx = np.arange(len(X_train))
        bloques = list(zip(np.array_split(X_train, size),
                           np.array_split(y_train, size),
                           np.array_split(idx, size)))
    else:
        bloques = None
    X_q, y_q, idx_q = comm.scatter(bloques, root=0)
    if fases: comm.Barrier()
    t2 = MPI.Wtime()

    # 3-4. Cada proceso calcula sus k vecinos locales (en paralelo)
    C_q = k_vecinos_locales(X_te, X_q, y_q, idx_q, k_l)
    t3 = MPI.Wtime()
    if fases: comm.Barrier()
    t4 = MPI.Wtime()

    # 5. Gather de los candidatos en el maestro
    C = comm.gather(C_q, root=0)
    t5 = MPI.Wtime()

    # 6-7. El maestro elige los k globales, vota y calcula la precision
    y_pred = acc = None
    if rank == 0:
        y_pred = mezclar_y_votar(C, k_l)
        acc = np.mean(y_pred == y_test)
    tf = MPI.Wtime()                     # tiempo final
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
        t.update(version=1, n_tr=len(X_train), n_te=len(X_test), p=size, rep=args.rep, modo=modo,
                 W=W, carga=carga_ini, hora=time.strftime("%Y-%m-%dT%H:%M:%S"))
        filas.append(t)
        preds[modo] = y_pred
    tiempos = "  ".join(f"T_ej({f['modo']}) = {f['T_ej']:.4f}" for f in filas)
    print(f"p = {size}  n_tr = {len(X_train)}  n_te = {len(X_test)}  Accuracy: {filas[0]['acc']:.4f}  "
          f"{tiempos} sec (beta 1, rep {args.rep}, W = {W}, cpu = {min(f['cpu'] for f in filas):.2f})")
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
