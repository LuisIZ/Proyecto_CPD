"""
Proyecto CS4052 - Caso 5 KNN - Grupo 7
Pruebas aparte que usamos en el analisis (datos originales, n_tr = 1437, n_te = 360):
    1. Empates de distancia en el k-esimo vecino.
    2. Costo por par del paso 3 (distancias con producto de matrices) y del paso 4 (ordenar).
    3. Contencion de hardware: cada proceso hace EL MISMO calculo completo (sin repartir);
       si el tiempo por copia sube con p, hay recursos compartidos (memoria, frecuencia).

Ejecucion:  for p in 1 4 8; do mpirun -np $p python pruebas_extra.py; done
"""
from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)
k = 3
REPS = 10


def paso3(Xb, X_q, norma_q):
    # distancias con ||x||^2 + ||y||^2 - 2 x.y (igual que la beta 3)
    D2 = (Xb ** 2).sum(axis=1)[:, None] + norma_q[None, :] - 2.0 * (Xb @ X_q.T)
    return np.sqrt(np.maximum(D2, 0.0))


def paso4(D):
    return np.argsort(D, axis=1, kind='stable')[:, :k]


norma = (X_train ** 2).sum(axis=1)
pares = len(X_test) * len(X_train)

if size == 1:
    # 1. empates en el k-esimo vecino (formula directa)
    D = np.sqrt(((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2))
    Ds = np.sort(D, axis=1)
    empates = int(np.sum(Ds[:, k - 1] == Ds[:, k]))
    print(f"Empates de distancia entre el vecino {k} y el {k + 1}: {empates} de {len(X_test)} puntos de prueba")

    # 2. costo por par de cada paso (un proceso)
    paso4(paso3(X_test, X_train, norma))
    t3 = t4 = 0.0
    for _ in range(REPS):
        t0 = MPI.Wtime()
        D = paso3(X_test, X_train, norma)
        t1 = MPI.Wtime()
        paso4(D)
        t2 = MPI.Wtime()
        t3 += t1 - t0
        t4 += t2 - t1
    ns3 = t3 / REPS / pares * 1e9
    ns4 = t4 / REPS / pares * 1e9
    print(f"Paso 3 (distancias): {ns3:.2f} ns por par")
    print(f"Paso 4 (ordenar):    {ns4:.2f} ns por par = {ns4 / np.log2(len(X_train)):.2f} ns por par y por nivel de log2(n_tr)")

# 3. contencion: p copias del mismo calculo completo al mismo tiempo
paso4(paso3(X_test, X_train, norma))
comm.Barrier()
t0 = MPI.Wtime()
for _ in range(REPS):
    paso4(paso3(X_test, X_train, norma))
t = (MPI.Wtime() - t0) / REPS
T = comm.gather(t, root=0)
if rank == 0:
    print(f"Contencion: p = {size} copias, tiempo por copia = {np.mean(T) * 1e3:.2f} ms (max {np.max(T) * 1e3:.2f} ms)")
