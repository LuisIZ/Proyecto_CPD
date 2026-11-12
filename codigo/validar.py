"""
Proyecto CS4052 - Caso 5 KNN - Grupo 7
Validacion: las predicciones de cada version tienen que ser identicas a las del codigo
base del profesor (knn_predict de knn_digits_sec.py, copiada en las betas) y a las de la
version secuencial de la beta 3. Cada caso corre la beta con --validar y
--calentamientos 0 (una corrida medida por modo; se comparan las de los dos modos).
    n0 (datos originales): betas 1, 2, 3 y 3b con p = 1 a 8, contra el codigo base
                           (y la precision tiene que ser 0,9833)
    n/4 y 4n:              betas 3 y 3b con p = 4 y 8, contra la secuencial y el codigo base
    16n:                   betas 3 y 3b con p = 8, contra la secuencial (con 1,3e8 pares el
                           bucle de Python del codigo base tardaria unos 4 min)
    debil (n ~ raiz(p)):   beta 3 con p = 2, 4 y 8 (cada tamano con su p), contra la
                           secuencial y el codigo base
    iso (n ~ k p^2):       n0, 4n y 16n ya estan arriba; el de 115 008 imagenes se omite
                           (la version secuencial sola tarda unos 3 min)
    relleno:               betas 3 y 3b con n_tr = 7 y n_te = 36, p = 4 (bloques de 2, 2, 2 y 1,
                           menos que k = 3) y p = 8 (un bloque vacio): prueba los candidatos +inf,
                           contra la secuencial y el codigo base

Ejecucion:  python validar.py resultados/validacion.txt
"""
import os
import re
import subprocess
import sys
import time
import numpy as np

PY = os.environ.get("PY", sys.executable)
CODIGO = os.path.dirname(os.path.abspath(__file__))
HILOS = ["OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
ENTORNO = dict(os.environ, **{h: "1" for h in HILOS})          # 1 hilo por proceso
MPIRUN = ["mpirun"] + [a for h in HILOS for a in ("-x", h)] + ["-np"]
SCRIPT = {"1": "knn_mpi_beta1.py", "2": "knn_mpi_beta2.py", "3": "knn_mpi_beta3.py", "3b": "knn_mpi_beta3.py"}

# casos: (tamano, version, factor de n_tr, factor de n_te, p)
CASOS = [("n0", v, 1, 1, p) for v in ["1", "2", "3", "3b"] for p in range(1, 9)]
CASOS += [(nombre, v, f, f, p) for nombre, f in [("n/4", 0.25), ("4n", 4)] for v in ["3", "3b"] for p in [4, 8]]
CASOS += [("16n", v, 16, 16, 8) for v in ["3", "3b"]]
CASOS += [(f"debil p={p}", "3", float(np.sqrt(p)), float(np.sqrt(p)), p) for p in [2, 4, 8]]
CASOS += [("relleno", v, 0.005, 0.1, p) for v in ["3", "3b"] for p in [4, 8]]   # bloques con menos de k puntos
OMITIDOS = ["iso p=8 (n = 115 008 imagenes, 2,1e9 pares): omitido, la version secuencial sola "
            "tarda unos 3 min en este Mac"]


def correr(cmd):
    r = subprocess.run(cmd, cwd=CODIGO, env=ENTORNO, capture_output=True, text=True, timeout=300)
    return r.stdout + r.stderr


salida = sys.argv[1] if len(sys.argv) > 1 else "resultados/validacion.txt"
os.makedirs(os.path.dirname(os.path.abspath(salida)), exist_ok=True)
versiones = correr([PY, "-c", "import sys, numpy, sklearn, mpi4py, matplotlib, pandas; from mpi4py import MPI; "
                    "print('Python', sys.version.split()[0], '| numpy', numpy.__version__, '| scikit-learn', "
                    "sklearn.__version__, '| mpi4py', mpi4py.__version__, '| matplotlib', matplotlib.__version__, "
                    "'| pandas', pandas.__version__); print(MPI.Get_library_version().splitlines()[0])"])
lineas = [f"== Validacion {time.strftime('%Y-%m-%d %H:%M:%S')}", versiones.replace("\x00", "").strip(),
          "Cada caso: la beta con --validar y --calentamientos 0; se comparan las predicciones de los modos A y B.",
          f"{'tamano':<11} {'version':<8} {'p':>2} {'n_tr':>6} {'n_te':>5} {'accuracy':>9}  "
          f"{'= secuencial':>12} {'= codigo base':>13}  resultado"]
print("\n".join(lineas), flush=True)
correctos = 0
for nombre, v, f_tr, f_te, p in CASOS:
    cmd = MPIRUN + [str(p), PY, SCRIPT[v], "--validar", "--calentamientos", "0"]
    if v in ("3", "3b"):
        cmd += ["--fac_tr", str(f_tr), "--fac_te", str(f_te)]
    if v == "3b":
        cmd.append("--reduccion")
    try:
        out = correr(cmd)
    except subprocess.TimeoutExpired:
        out = "se paso del tiempo maximo"
    n = re.search(r"n_tr = (\d+)\s+n_te = (\d+)\s+Accuracy: ([\d.]+)", out)
    sec = re.search(r"iguales a la version secuencial: (True|False)", out)
    base = re.search(r"iguales al codigo base: (True|False)", out)
    comparaciones = [m.group(1) == "True" for m in (sec, base) if m]
    ok = n is not None and len(comparaciones) > 0 and all(comparaciones)
    if nombre == "n0":                   # con los datos originales la precision es la del codigo base
        ok = ok and n.group(3) == "0.9833"
    if v in ("3", "3b") and (sec is None or (base is None and "omitida" not in out)):
        ok = False                       # la beta 3 siempre compara con la secuencial
    correctos += ok
    linea = (f"{nombre:<11} {'beta ' + v:<8} {p:>2} {n.group(1) if n else '?':>6} {n.group(2) if n else '?':>5} "
             f"{n.group(3) if n else '?':>9}  {sec.group(1) if sec else '--':>12} "
             f"{base.group(1) if base else ('omitida' if 'omitida' in out else '--'):>13}  "
             f"{'OK' if ok else 'REVISAR'}")
    if not ok:
        linea += "\n    " + out.strip().replace("\n", "\n    ")[-1500:]
    lineas.append(linea)
    print(linea, flush=True)
lineas += [f"(omitido) {o}" for o in OMITIDOS]
lineas.append(f"Resumen: {correctos} de {len(CASOS)} casos correctos "
              f"({'todas las predicciones son identicas' if correctos == len(CASOS) else 'hay casos para revisar'})")
print("\n".join(lineas[-1 - len(OMITIDOS):]), flush=True)
with open(salida, "w") as f:
    f.write("\n".join(lineas) + "\n")
