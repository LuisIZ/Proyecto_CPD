"""Funciones compartidas por las versiones beta del KNN en paralelo."""
import csv
import os
import time
from collections import Counter

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

K = 3
N_BASE = 1797
SEMILLA = 42

COLUMNAS = [
    "version", "n", "p", "repeticion", "n_tr", "n_te",
    "t_total", "t_comp", "t_comp_paralelo", "t_dist", "t_sort", "t_seleccion",
    "t_comm", "t_comm_max", "t_bcast", "t_scatter", "t_recoleccion", "flop", "accuracy",
]


def cargar_datos(n=N_BASE):
    """Devuelve X_tr, X_te, y_tr, y_te con n muestras en total.

    Con n = 1797 es exactamente la división del código base. Con n menor se
    toma una submuestra estratificada. Con n mayor se replican las imágenes con
    ruido entero, recortado al rango 0-16 de los píxeles: esos datos solo sirven
    para medir tiempos, y su precisión no es comparable con la del código base.
    """
    digitos = load_digits()
    X, y = digitos.data, digitos.target
    rng = np.random.default_rng(SEMILLA)
    if n < len(X):
        X, _, y, _ = train_test_split(X, y, train_size=n, stratify=y,
                                      random_state=SEMILLA)
    elif n > len(X):
        copias = -(-n // len(X))
        X = np.tile(X, (copias, 1))[:n]
        y = np.tile(y, copias)[:n]
        X = np.clip(X + rng.integers(-1, 2, X.shape), 0.0, 16.0)
    return train_test_split(X, y, test_size=0.2, random_state=SEMILLA)


def repartir(X_tr, y_tr, p):
    """Arma los p bloques para comm.scatter.

    Los primeros n_tr mod p bloques tienen un punto más. Cada bloque lleva los
    índices globales de sus puntos, que se usan para desempatar distancias.
    """
    bloques = np.array_split(np.arange(len(X_tr)), p)
    return [(X_tr[b], y_tr[b], b) for b in bloques]


def vecinos_locales(X_te, X_b, y_b, i_b, k, reloj=time.perf_counter):
    """Pasos 3 y 4: los k vecinos de cada punto de prueba dentro del bloque.

    Devuelve (dist, etiquetas, indices), cada uno de forma (n_te, k), y el
    tiempo usado en distancias y en ordenamiento. Si el bloque tiene menos de
    k puntos, el resto queda con distancia infinita.

    Las diferencias se calculan en un buffer reutilizado. Con (X_b - x) ** 2,
    NumPy pide memoria nueva en cada iteración y, en bloques mayores a ~0,7 MB,
    cada distancia cuesta unas 4 veces más. Eso penaliza a los valores pequeños
    de p e infla el speedup de forma artificial.
    """
    n_te = len(X_te)
    m = min(k, len(X_b))
    dist = np.full((n_te, k), np.inf)
    etiquetas = np.full((n_te, k), -1, dtype=np.int64)
    indices = np.full((n_te, k), -1, dtype=np.int64)
    buffer = np.empty_like(X_b)
    t_dist = t_sort = 0.0
    for i in range(n_te):
        t0 = reloj()
        np.subtract(X_b, X_te[i], out=buffer)
        np.multiply(buffer, buffer, out=buffer)
        D = np.sqrt(buffer.sum(axis=1))
        t1 = reloj()
        orden = np.argsort(D, kind="stable")[:m]
        dist[i, :m] = D[orden]
        etiquetas[i, :m] = y_b[orden]
        indices[i, :m] = i_b[orden]
        t2 = reloj()
        t_dist += t1 - t0
        t_sort += t2 - t1
    return (dist, etiquetas, indices), t_dist, t_sort


def seleccionar_k(dist, etiquetas, indices, k):
    """Paso 6: los k pares de menor distancia por fila, ordenados.

    Los empates se resuelven por índice global, para que el resultado no
    dependa de cómo se repartieron los datos.
    """
    orden = np.lexsort((indices, dist))[:, :k]
    return (
        np.take_along_axis(dist, orden, axis=1),
        np.take_along_axis(etiquetas, orden, axis=1),
        np.take_along_axis(indices, orden, axis=1),
    )


def combinar(candidatos, k):
    """Une las listas de candidatos de todos los procesos y elige los k mejores."""
    unidos = [np.concatenate(partes, axis=1) for partes in zip(*candidatos)]
    return seleccionar_k(*unidos, k)


def votar(etiquetas):
    """Votación por mayoría. Las etiquetas llegan de menor a mayor distancia.

    Counter resuelve los empates de votos por orden de aparición, igual que el
    código base.
    """
    return np.array([Counter(fila.tolist()).most_common(1)[0][0] for fila in etiquetas])


def armar_fila(version, args, p, n_tr, medida, por_proceso):
    """Combina los tiempos del maestro con los de todos los procesos."""
    comp, dist, sort, flop, comm = zip(*por_proceso)
    return {
        "version": version, "n": args.n, "p": p, "repeticion": args.repeticion,
        "n_tr": n_tr, "n_te": medida["n_te"],
        "t_total": medida["t_total"],
        "t_comp": max(comp) + medida["t_seleccion"],
        "t_comp_paralelo": max(comp), "t_dist": max(dist), "t_sort": max(sort),
        "t_seleccion": medida["t_seleccion"],
        "t_comm": medida["t_bcast"] + medida["t_scatter"] + medida["t_recoleccion"],
        "t_comm_max": max(comm),
        "t_bcast": medida["t_bcast"], "t_scatter": medida["t_scatter"],
        "t_recoleccion": medida["t_recoleccion"],
        "flop": sum(flop), "accuracy": medida["accuracy"],
    }


def guardar_csv(ruta, fila):
    """Agrega una fila de mediciones al CSV."""
    nuevo = not os.path.exists(ruta)
    with open(ruta, "a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS)
        if nuevo:
            escritor.writeheader()
        escritor.writerow(fila)
