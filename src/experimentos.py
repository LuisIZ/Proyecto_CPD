"""Ejecuta la malla de experimentos y guarda las mediciones en un CSV.

Cada medición usa un mpiexec separado, así las repeticiones no necesitan
sincronizarse. La repetición es el bucle externo, para repartir entre todas las
configuraciones cualquier cambio lento de la máquina, como la temperatura.
Si se interrumpe, al volver a ejecutarlo continúa con lo que falta.

Uso: python experimentos.py [--repeticiones 5] [--salida ../results/final]
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from lanzador import RAIZ, ejecutar, ejecutar_sin_mpi, entorno

VALORES_P = [1, 2, 4, 8, 16]
VALORES_N = [450, 900, 1350, 1797, 3600, 7200]
VERSIONES = {"gather": "beta2_knn_mpi.py", "reduccion": "beta3_knn_mpi.py"}


def ya_medidas(ruta):
    """Configuraciones que ya están en el CSV."""
    if not ruta.exists():
        return set()
    with open(ruta, encoding="utf-8") as archivo:
        return {(f["version"], int(f["n"]), int(f["p"]), int(f["repeticion"]))
                for f in csv.DictReader(archivo)}


def ya_secuenciales(ruta):
    """Corridas de T_s que ya están en el CSV."""
    if not ruta.exists():
        return set()
    with open(ruta, encoding="utf-8") as archivo:
        return {(int(f["n"]), int(f["repeticion"])) for f in csv.DictReader(archivo)}


def medir_base(ruta, repeticiones):
    """Mide el código base sin modificar, para reportarlo junto a T_s."""
    hechas = 0
    if ruta.exists():
        hechas = sum(1 for _ in open(ruta, encoding="utf-8")) - 1
    for repeticion in range(hechas, repeticiones):
        env = entorno()
        env["MPLBACKEND"] = "Agg"
        salida = subprocess.run(
            [sys.executable, str(RAIZ / "drive" / "knn_digits_sec.py")],
            env=env, text=True, encoding="utf-8", capture_output=True, check=True).stdout
        tiempo = re.search(r"Execution time \(sequential\): ([\d.]+)", salida).group(1)
        precision = re.search(r"Accuracy: ([\d.]+)", salida).group(1)
        nuevo = not ruta.exists()
        with open(ruta, "a", encoding="utf-8") as archivo:
            if nuevo:
                archivo.write("repeticion,t_total,accuracy\n")
            archivo.write(f"{repeticion},{tiempo},{precision}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--salida", default=str(RAIZ / "results" / "final"))
    parser.add_argument("--p", type=int, nargs="+", default=VALORES_P)
    parser.add_argument("--n", type=int, nargs="+", default=VALORES_N)
    args = parser.parse_args()

    carpeta = Path(args.salida)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "mediciones.csv"
    hechas = ya_medidas(ruta)

    ruta_pingpong = carpeta / "pingpong.csv"
    if not ruta_pingpong.exists():
        print("Midiendo a y b con el ping-pong...", flush=True)
        ejecutar("ping_pong.py", 2, "--csv", ruta_pingpong)

    print("Midiendo el código base...", flush=True)
    medir_base(carpeta / "base.csv", args.repeticiones)

    ruta_secuencial = carpeta / "secuencial.csv"

    total = args.repeticiones * len(VERSIONES) * len(args.p) * len(args.n)
    contador, inicio = 0, time.time()
    for repeticion in range(args.repeticiones):
        for n in args.n:
            for intento in range(1, 4):
                if (n, repeticion) in ya_secuenciales(ruta_secuencial):
                    break
                try:
                    ejecutar_sin_mpi("secuencial_vectorizado.py", "--n", n,
                                     "--repeticion", repeticion, "--csv", ruta_secuencial,
                                     tiempo_max=300)
                except subprocess.TimeoutExpired:
                    # A veces se cuelga al terminar, después de escribir su fila.
                    print(f"AVISO: secuencial_vectorizado.py con n={n} se colgó "
                          f"(intento {intento}).", flush=True)
            for p in args.p:
                for version, script in VERSIONES.items():
                    contador += 1
                    if (version, n, p, repeticion) in hechas:
                        continue
                    salida = ejecutar(script, p, "--n", n, "--repeticion", repeticion,
                                      "--csv", ruta)
                    minutos = (time.time() - inicio) / 60
                    print(f"[{contador}/{total}] {minutos:5.1f} min {version:<10}"
                          f"{salida.strip()}", flush=True)


if __name__ == "__main__":
    main()
