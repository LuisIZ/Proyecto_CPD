"""
Proyecto CS4052 - Computacion Paralela y Distribuida (UTEC 2026-II)
Caso 5: KNN en paralelo con MPI (mpi4py) - Grupo 7

Version beta 3 (version final; cambios respecto a la beta 2):
    - Distancias por bloques de puntos de prueba: la matriz de distancias de un
      bloque se calcula con ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
      (es la misma distancia euclidiana, pero con un producto de matrices).
    - Tamano del problema variable (inciso c y e del caso 5):
        --fac 0.25 | 1 | 4 | 16   tamano relativo al load_digits original
        --fac_tr, --fac_te        factores de n_tr y n_te por separado (los usa el driver)
        --ley fuerte | debil_n | iso_n | control
            fuerte:  n fijo (definido por --fac)
            debil_n: n = n0*raiz(p)   (un solo n: trabajo por proceso constante)
            iso_n:   n = n0*p^2       (un solo n: isoeficiencia)
            control: n_tr = 1437*p, n_te = 360   (n_te fijo; serie de control)
      Los datos mas grandes son replicas con ruido gaussiano (sigma = 1 pixel)
      hechas despues del split, sin mezclar entrenamiento con prueba.
    - --secuencial: la misma version sin comunicacion, para medir T_s.
    - --reduccion (beta 3b, mejora): el gather de los candidatos y la mezcla del maestro
      se cambian por comm.reduce(C_q, op=mezcla, root=0) y el maestro solo vota. La
      version principal sigue siendo la del gather, que es la que pide el enunciado.
    - Si un bloque tiene menos de k puntos, los candidatos que faltan quedan con
      distancia +inf y no ganan nunca.
    - Protocolo de medicion (el mismo en las tres betas): cada ejecucion mide UNA
      repeticion (--rep r). Modo A: sin barreras dentro de la region medida (de ahi sale
      T_ej). Modo B: con comm.Barrier() antes de cada marca, solo para separar las fases.
      Primero hay W rondas de calentamiento que se descartan; en cada ronda se corre una
      vez cada modo, asi los dos llegan igual de "calientes" a la medicion (las primeras
      corridas de un proceso son mas lentas: hasta 2 veces en n/4 con p = 8). Despues va
      la ronda medida: una corrida por modo. El orden de los modos alterna con la
      repeticion: A,B si r es par y B,A si es impar.
      W por defecto, por tiempo: el maestro cronometra la primera ronda (d por corrida) y
      W = techo(0,25 s / d), entre 1 y 30; W = 0 si n_tr*n_te > 2e8 (--calentamientos lo fija).
      Columna cpu: minimo entre procesos del tiempo de CPU / tiempo de pared en la corrida
      medida (~1 si ningun proceso fue desalojado por otro programa).

Ejecucion:
    mpirun -np 4 python knn_mpi_beta3.py --fac 4 --rep 0 --csv resultados/tiempos.csv
    mpirun -np 4 python knn_mpi_beta3.py --fac 4 --reduccion              (beta 3b)
    python knn_mpi_beta3.py --secuencial --fac 4 --rep 0 --csv resultados/tiempos.csv
    mpirun -np 4 python knn_mpi_beta3.py --validar
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


def replicar(X, y, f, semilla):
    # f < 1: subconjunto; f >= 1: copias con ruido gaussiano (f puede ser fraccionario:
    # las copias completas son int(f) y de la ultima se toma solo la parte que falta)
    if f < 1:
        m = int(f * len(X))
        return X[:m], y[:m]
    rng = np.random.default_rng(semilla)
    n = len(X)
    total = int(round(f * n))
    Xs, ys = [X], [y]
    while sum(len(a) for a in Xs) < total:
        faltan = total - sum(len(a) for a in Xs)
        copia = np.clip(X + rng.normal(0.0, 1.0, X.shape), 0, 16)
        Xs.append(copia[:faltan])
        ys.append(y[:faltan])
    return np.vstack(Xs), np.concatenate(ys)


def generar_datos(f_tr, f_te):
    # mismo split del codigo base; despues se agranda (o se reduce) cada parte
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )
    X_train, y_train = replicar(X_train, y_train, f_tr, semilla=0)
    X_test, y_test = replicar(X_test, y_test, f_te, semilla=1)
    return X_train, X_test, y_train, y_test


def knn_predict(test_point, X_train, y_train, k):
    # funcion del codigo base (solo se usa para validar)
    distances = [np.sqrt(np.sum((test_point - x) ** 2)) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    return Counter(k_labels).most_common(1)[0][0]


def k_vecinos_locales(X_test, X_q, y_q, idx_q, k):
    # pasos 3 y 4: matriz de distancias por bloques de B puntos de prueba.
    # Si el bloque tiene menos de k puntos, los candidatos que faltan quedan con
    # distancia +inf (etiqueta e indice -1) y no ganan nunca en la mezcla.
    n_te, n_q = len(X_test), len(X_q)
    m = min(k, n_q)
    dist = np.full((n_te, k), np.inf)
    etiq = np.full((n_te, k), -1, dtype=int)
    glob = np.full((n_te, k), -1, dtype=int)
    norma_q = (X_q ** 2).sum(axis=1)
    B = max(1, min(512, 2 ** 22 // max(n_q, 1)))   # bloque de a lo mas ~32 MB
    for s in range(0, n_te, B):
        Xb = X_test[s:s + B]
        # paso 3: distancia euclidiana (se cuentan 3 FLOP por caracteristica)
        D2 = (Xb ** 2).sum(axis=1)[:, None] + norma_q[None, :] - 2.0 * (Xb @ X_q.T)
        D = np.sqrt(np.maximum(D2, 0.0))
        # paso 4: ordenar cada fila y tomar los k menores
        I = np.argsort(D, axis=1, kind='stable')[:, :m]
        dist[s:s + B, :m] = np.take_along_axis(D, I, axis=1)
        etiq[s:s + B, :m] = y_q[I]
        glob[s:s + B, :m] = idx_q[I]
    return dist, etiq, glob


def votar(L):
    # votacion: gana la etiqueta mas repetida; si hay empate, la del vecino mas cercano
    # (L tiene las k etiquetas de cada punto de prueba, de menor a mayor distancia)
    cuenta = (L[:, :, None] == L[:, None, :]).sum(axis=2)
    return L[np.arange(len(L)), np.argmax(cuenta, axis=1)]


def mezclar_y_votar(C, k):
    # paso 6: juntar los p*k candidatos de cada punto de prueba
    dist = np.hstack([c[0] for c in C])
    etiq = np.hstack([c[1] for c in C])
    glob = np.hstack([c[2] for c in C])
    orden = np.lexsort((glob, dist), axis=1)          # por distancia y luego indice global
    L = np.take_along_axis(etiq, orden, axis=1)[:, :k]
    return votar(L)


def mezcla(a, b):
    # beta 3b: operacion de comm.reduce. Por cada punto de prueba deja los k candidatos de
    # menor (distancia, indice global) de la union de a y b, ordenados con el mismo criterio
    # de mezclar_y_votar, asi que las predicciones son identicas a las del gather.
    # Es asociativa y conmutativa: los k menores de una union no dependen del orden ni de
    # como se agrupen las listas (el indice global desempata y no se repite; los rellenos
    # con +inf son todos iguales).
    k = a[0].shape[1]
    dist = np.hstack((a[0], b[0]))
    etiq = np.hstack((a[1], b[1]))
    glob = np.hstack((a[2], b[2]))
    orden = np.lexsort((glob, dist), axis=1)[:, :k]
    return (np.take_along_axis(dist, orden, axis=1), np.take_along_axis(etiq, orden, axis=1),
            np.take_along_axis(glob, orden, axis=1))


def una_corrida(X_train, y_train, X_test, y_test, k, fases, reduccion):
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
    if reduccion:
        C = comm.reduce(C_q, op=mezcla, root=0)       # 5-6. (3b) reduccion: k globales en el maestro
    else:
        C = comm.gather(C_q, root=0)                  # 5. Gather de los candidatos
    t5 = MPI.Wtime()
    y_pred = acc = None
    if rank == 0:
        if reduccion:
            y_pred = votar(C[1])                      # 6. (3b) solo la votacion
        else:
            y_pred = mezclar_y_votar(C, k_l)          # 6. k globales y votacion
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


def corrida_secuencial(X_train, y_train, X_test, y_test, k):
    # misma version pero en un solo proceso y sin comunicacion (para T_s)
    c0 = time.process_time()
    t0 = MPI.Wtime()
    C = [k_vecinos_locales(X_test, X_train, y_train, np.arange(len(X_train)), k)]
    y_pred = mezclar_y_votar(C, k)
    acc = np.mean(y_pred == y_test)
    tf = MPI.Wtime()
    c1 = time.process_time()
    return dict(T_ej=tf - t0, T_bcast=0, T_scatter=0, T_comp_max=0, T_comp_prom=0,
                T_gather=0, T_comp2=0, acc=acc, cpu=(c1 - c0) / (tf - t0)), y_pred


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
parser.add_argument("--fac", type=float, default=1)
parser.add_argument("--fac_tr", type=float, default=None)   # factor de n_tr (si no, el de --fac o --ley)
parser.add_argument("--fac_te", type=float, default=None)   # factor de n_te (si no, el de --fac o --ley)
parser.add_argument("--ley", default="fuerte", choices=["fuerte", "debil_n", "iso_n", "control"])
parser.add_argument("--p_ley", type=int, default=1)         # p de la ley (solo con --secuencial)
parser.add_argument("--rep", type=int, default=0)           # indice de la repeticion (lo pone el driver)
parser.add_argument("--calentamientos", type=int, default=None)   # rondas W (si no, por tiempo)
parser.add_argument("--carga", type=float, default=None)    # carga externa (la mide el driver)
parser.add_argument("--csv", default="")
parser.add_argument("--validar", action="store_true")
parser.add_argument("--secuencial", action="store_true")
parser.add_argument("--reduccion", action="store_true")     # beta 3b: comm.reduce en vez de gather
args = parser.parse_args()

# factores de tamano segun la ley elegida
p_ley = args.p_ley if args.secuencial else size
if args.ley == "fuerte":
    f_tr = f_te = args.fac
elif args.ley == "control":              # n_tr = 1437 p, n_te = 360 (n_te fijo)
    f_tr, f_te = p_ley, 1
elif args.ley == "debil_n":              # un solo n: trabajo por proceso constante, n ~ raiz(p)
    f_tr = f_te = float(np.sqrt(p_ley))
else:                                    # iso_n: un solo n, isoeficiencia n ~ p^2
    f_tr = f_te = float(p_ley ** 2)
if args.fac_tr is not None:
    f_tr = args.fac_tr
if args.fac_te is not None:
    f_te = args.fac_te
version = "3b" if args.reduccion and not args.secuencial else "3"

# 0. Datos iniciales en el maestro
if rank == 0:
    X_train, X_test, y_train, y_test = generar_datos(f_tr, f_te)
    k = 3
    pares = len(X_train) * len(X_test)
    carga_ini = args.carga if args.carga is not None else carga()
else:
    X_train = y_train = X_test = y_test = k = pares = None
pares = comm.bcast(pares, root=0)        # todos hacen las mismas corridas (fuera de la medicion)

if args.secuencial:
    modos = [("S", None)]
else:
    modos = [("A", False), ("B", True)]
    if args.rep % 2 == 1:
        modos.reverse()                  # repeticiones impares: primero B y luego A


def ronda():
    # una corrida de cada modo, en el orden de la repeticion: [(modo, tiempos, y_pred)] en el maestro
    salida = []
    for modo, fases in modos:
        if args.secuencial:
            t, y_pred = corrida_secuencial(X_train, y_train, X_test, y_test, k)
        else:
            t, y_pred = una_corrida(X_train, y_train, X_test, y_test, k, fases, args.reduccion)
        salida.append((modo, t, y_pred))
    return salida


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
        t.update(version=version, n_tr=len(X_train), n_te=len(X_test), p=1 if args.secuencial else size,
                 rep=args.rep, modo=modo, W=W, carga=carga_ini, hora=time.strftime("%Y-%m-%dT%H:%M:%S"))
        filas.append(t)
        preds[modo] = y_pred
    tiempos = "  ".join(f"T_ej({f['modo']}) = {f['T_ej']:.4f}" for f in filas)
    print(f"p = {size}  n_tr = {len(X_train)}  n_te = {len(X_test)}  Accuracy: {filas[0]['acc']:.4f}  "
          f"{tiempos} sec (beta {version}{', secuencial' if args.secuencial else ''}, rep {args.rep}, W = {W}, "
          f"cpu = {min(f['cpu'] for f in filas):.2f})")
    if args.csv:
        nuevo = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
        with open(args.csv, "a") as f:
            if nuevo:
                f.write(",".join(COLUMNAS) + "\n")
            for fila in filas:
                f.write(",".join(str(fila[c]) for c in COLUMNAS) + "\n")
    if args.validar:
        # validacion 1: mismas predicciones (de cada modo) que la version secuencial del mismo calculo
        _, y_s = corrida_secuencial(X_train, y_train, X_test, y_test, k)
        print(f"Predicciones iguales a la version secuencial: {all(np.array_equal(y, y_s) for y in preds.values())}")
        # validacion 2: igual al codigo del profesor (knn_predict); con mas de 1e7 pares el
        # bucle de Python tardaria minutos y ahi solo se compara con la version secuencial
        pares = len(X_train) * len(X_test)
        if pares <= 1e7:
            y_sec = np.array([knn_predict(x, X_train, y_train, k) for x in X_test])
            print(f"Predicciones iguales al codigo base: {all(np.array_equal(y, y_sec) for y in preds.values())}")
        else:
            print(f"Comparacion con el codigo base omitida: {pares:.2e} pares (> 1e7)")
