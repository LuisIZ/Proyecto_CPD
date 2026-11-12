"""Ejecuta los scripts de MPI desde validar_betas.py y experimentos.py."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

CARPETA = Path(__file__).resolve().parent
RAIZ = CARPETA.parent


def buscar_mpiexec():
    """Usa el mpiexec del entorno virtual (impi-rt) o, si no hay, el del sistema."""
    propio = Path(sys.prefix) / "Library" / "bin" / "mpiexec.exe"
    return str(propio) if propio.exists() else shutil.which("mpiexec")


def entorno():
    """Variables de entorno para medir con un solo hilo por proceso."""
    env = dict(os.environ)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[variable] = "1"
    biblioteca = Path(sys.prefix) / "Library"
    if (biblioteca / "bin" / "mpiexec.exe").exists():
        env["I_MPI_ROOT"] = str(biblioteca)
        env["PATH"] = str(biblioteca / "bin") + os.pathsep + env["PATH"]
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def ejecutar_sin_mpi(script, *argumentos, tiempo_max=900):
    """Lanza un script en un solo proceso, sin mpiexec."""
    comando = [sys.executable, str(CARPETA / script)] + [str(a) for a in argumentos]
    return subprocess.run(comando, env=entorno(), cwd=CARPETA, text=True, encoding="utf-8",
                          capture_output=True, timeout=tiempo_max, check=True).stdout


def terminar(proceso):
    """Termina mpiexec y los procesos que lanzó."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proceso.pid)], capture_output=True)
    else:
        proceso.kill()
    proceso.communicate()


def ejecutar(script, p, *argumentos, tiempo_max=120, intentos=3):
    """Lanza un script con p procesos y devuelve su salida.

    En raras ocasiones una corrida de mpiexec se queda colgada. Si pasa, se
    descarta y se repite, para no perder el resto de la malla.
    """
    comando = [buscar_mpiexec(), "-n", str(p), sys.executable, str(CARPETA / script)]
    comando += [str(a) for a in argumentos]
    for intento in range(1, intentos + 1):
        proceso = subprocess.Popen(comando, env=entorno(), cwd=CARPETA, text=True,
                                   encoding="utf-8", stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        try:
            salida, errores = proceso.communicate(timeout=tiempo_max)
        except subprocess.TimeoutExpired:
            terminar(proceso)
            print(f"AVISO: {script} con p={p} {argumentos} se colgó (intento {intento}).",
                  flush=True)
            continue
        if proceso.returncode != 0:
            raise RuntimeError(f"Falló {script} con p={p}:\n{errores[-2000:]}")
        return salida
    raise RuntimeError(f"{script} con p={p} se colgó {intentos} veces.")
