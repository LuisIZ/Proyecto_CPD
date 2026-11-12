"""
Proyecto CS4052 - Caso 5 KNN - Grupo 7
Barrido de experimentos. Correr con la maquina libre (sin OneDrive, navegador, etc.) y enchufada:
    caffeinate -i python correr_experimentos.py                 # carpeta resultados_nuevos, 4 repeticiones
    caffeinate -i python correr_experimentos.py otra_carpeta --reps 6
    python correr_experimentos.py prueba_rapida --rapido        # malla chica, para probar (unos minutos)
    python correr_experimentos.py --plan                        # solo muestra la malla y cuanto tardaria

Protocolo de medicion (aporte de Luis, con los ajustes de la revision):
    - Cada mpirun mide UNA repeticion de UNA configuracion (version, n_tr, n_te, p): W rondas de
      calentamiento con los modos intercalados, que se descartan, y una corrida medida por modo
      (ver knn_mpi_beta3.py).
    - Las repeticiones van afuera: primero la repeticion 0 de toda la malla, despues la 1, etc.
      En las repeticiones pares la malla se recorre hacia adelante y en las impares hacia atras.
      Con un numero PAR de repeticiones, la mediana de cada configuracion cae en el mismo punto
      medio de la sesion para todas (una deriva lineal de la maquina se cancela) y cada modo va
      primero en la mitad de las repeticiones; con un numero impar la mediana es la repeticion del
      medio y la deriva no se cancela. Por eso son 4 (y 2 para las de mas de 1e9 pares).
    - Cada configuracion se mide una vez por repeticion aunque este en varias series (por
      ejemplo n0 con p = 1 esta en la fuerte, la debil, la iso y la de control): analisis.py
      arma las series buscando por tamano, y todas comparten la misma linea base. El T_s de cada
      tamano va junto a sus corridas en paralelo.
    - Antes de cada mpirun se mide la carga externa (nucleos que usan los procesos ajenos al
      barrido) y se pasa a la beta con --carga. Despues de una corrida de mas de 20 s hay una pausa
      de 3 s.
    - Cada mpirun tiene un tiempo maximo (5 veces lo estimado, al menos 60 s) y un reintento; los
      fallos van a fallos.log. Si se pasa del tiempo, se matan mpirun y todos sus procesos.
    - Si la carpeta ya tiene una corrida completa no la pisa; si tiene una parcial, la retoma:
      salta las (version, n_tr, n_te, p, rep) que ya estan en tiempos.csv, y si una quedo a medias
      (por ejemplo, solo con el modo B) quita sus filas y la vuelve a medir. No retoma un tiempos.csv
      que no sea de este barrido (otras columnas o sin malla.json, como el de septiembre).
    - En macOS el driver se mantiene despierto con caffeinate -i -w y anota en entorno.txt, al
      inicio y al final, si la Mac estaba enchufada, el modo de bajo consumo y el estado termico.
Paso 0: validacion (validacion.txt), ping-pong (pingpong.csv) y pruebas_extra.py con 1, 4 y 8
copias (pruebas_extra.txt). Al inicio se escribe el entorno (entorno.txt).
"""
import argparse
import csv
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time

CODIGO = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("PY", sys.executable)
HILOS = ["OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
ENTORNO = dict(os.environ, MPLBACKEND="Agg", **{h: "1" for h in HILOS})   # 1 hilo por proceso
MPIRUN = ["mpirun"] + [a for h in HILOS for a in ("-x", h)] + ["-np"]
COLUMNAS = ["version", "n_tr", "n_te", "p", "rep", "modo", "W", "T_ej", "T_bcast", "T_scatter",
            "T_comp_max", "T_comp_prom", "T_gather", "T_comp2", "acc", "carga", "cpu", "hora"]
N_TR, N_TE = 1437, 360     # tamano del split original (n0)
REPS, REPS_GRANDES = 4, 2  # repeticiones por defecto (pares) y las de las configuraciones de mas de 1e9 pares
ARRANQUE = 1.6             # s que tarda un mpirun en arrancar (importar numpy, sklearn y mpi4py)
COSTO = {"0": 2.0e-6, "1": 2.0e-6, "2": 1.0e-7}   # s por par con p = 1 (solo para el ETA)
T_CAL, W_MAX = 0.25, 30    # mismo calentamiento por tiempo que las betas (para el ETA)
PAUSA, LARGA = 3, 20       # s de pausa despues de un mpirun de mas de LARGA s
CARGA_ALTA = 1.5           # nucleos ocupados por otros programas: mas que esto, se avisa


def tamano(f, n):
    # mismo tamano que replicar() de knn_mpi_beta3.py
    return int(f * n) if f < 1 else int(round(f * n))


def conf(version, f_tr, f_te, p, secuencial=False):
    return dict(version=version, f_tr=f_tr, f_te=f_te, p=p, n_tr=tamano(f_tr, N_TR), n_te=tamano(f_te, N_TE),
                modos=("S",) if secuencial or version == "0" else ("A", "B"))


def sec(f):
    return conf("3", f, f, 1, secuencial=True)


def armar_malla(rapido):
    # configuraciones distintas; cada una se mide una vez por repeticion. El T_s (modo S) de cada
    # tamano va justo antes de sus corridas en paralelo
    if rapido:                           # prueba corta: n/4 y n0, p = 1, 2, 4, betas 3 y 3b y el secuencial
        return [c for f in (0.25, 1) for c in [sec(f)] + [conf(v, f, f, p) for p in (1, 2, 4) for v in ("3", "3b")]]
    P = [1, 2, 4, 6, 8]
    malla = [conf("0", 1, 1, 1)]                                       # codigo base del profesor (T_base)
    malla += [conf(v, 1, 1, p) for v in ("1", "2") for p in P]         # betas 1 y 2 con n0
    for f in (0.25, 1, 4, 16):                                         # fuerte
        malla += [sec(f)] + [conf(v, f, f, p) for p in P for v in ("3", "3b")]
    for p in (2, 4, 8):                                                # debil: n ~ raiz(p) (p = 1 es n0)
        malla += [sec(math.sqrt(p)), conf("3", math.sqrt(p), math.sqrt(p), p)]
    malla += [conf("3", p, 1, p) for p in (2, 4, 6, 8)]                # control: n_te = 360, n_tr = 1437 p
    malla += [sec(64), conf("3", 64, 64, 8)]   # iso: n ~ k p^2; con p = 1, 2, 4 coincide con n0, 4n y 16n
    return malla


def reps_de(c, reps):
    # las configuraciones con mas de 1e9 pares (115 008 imagenes) llevan menos repeticiones (tambien par)
    return min(reps, REPS_GRANDES) if c["n_tr"] * c["n_te"] > 1e9 else reps


def nombre(c):
    v = "codigo base" if c["version"] == "0" else f"beta {c['version']}" + (" sec" if c["modos"] == ("S",) else "")
    return f"{v:<12} n_tr = {c['n_tr']:>6}  n_te = {c['n_te']:>5}  p = {c['p']}"


def comando(c, rep, ruta_csv, carga):
    v = c["version"]
    if v == "0":
        return [PY, "knn_digits_sec.py"]
    tam = ["--fac_tr", str(c["f_tr"]), "--fac_te", str(c["f_te"])]
    comun = ["--rep", str(rep), "--csv", ruta_csv, "--carga", f"{carga:.3f}"]
    if c["modos"] == ("S",):
        return [PY, "knn_mpi_beta3.py", "--secuencial"] + tam + comun
    script = {"1": "knn_mpi_beta1.py", "2": "knn_mpi_beta2.py"}.get(v, "knn_mpi_beta3.py")
    extra = (tam + (["--reduccion"] if v == "3b" else [])) if v in ("3", "3b") else []
    return MPIRUN + [str(c["p"]), PY, script] + comun + extra


def estimar(c):
    # duracion a priori de un mpirun (s), para el ETA y el tiempo maximo: arranque + (W + 1) rondas con una
    # corrida por modo. En la beta 3 el costo por par crece con log2(n_tr/p), por el ordenamiento (~5 ns por
    # nivel); con p procesos a la vez cada uno va algo mas lento (memoria compartida): ~4 % por proceso extra
    pares = c["n_tr"] * c["n_te"]
    if c["version"] == "0":
        return ARRANQUE + 1.0 + pares * COSTO["0"]     # el codigo base ademas importa matplotlib
    if c["version"] in ("3", "3b"):
        costo = 5e-9 * math.log2(max(c["n_tr"] / c["p"], 2))
    else:
        costo = COSTO[c["version"]]
    d = pares * costo / c["p"] * (1 + 0.04 * (c["p"] - 1)) + 3e-4 * (c["p"] > 1)   # una corrida
    W = 0 if pares > 2e8 else min(W_MAX, max(1, math.ceil(T_CAL / d)))
    return ARRANQUE + (W + 1) * len(c["modos"]) * d


def tiempo_max(c):
    return max(60.0, 5 * estimar(c))


def ahora():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def carga():
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return float("nan")


def procesos():
    # [(pid, ppid, %cpu, nombre)] de todos los procesos (vacio si no hay ps, por ejemplo en Windows)
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,%cpu=,comm="], capture_output=True, text=True,
                             timeout=20, env=dict(os.environ, LC_ALL="C")).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    filas = []
    for linea in out.splitlines():
        partes = linea.split(None, 3)
        try:
            filas.append((int(partes[0]), int(partes[1]), float(partes[2]), partes[3] if len(partes) > 3 else ""))
        except (ValueError, IndexError):
            pass
    return filas


def descendientes(pid, filas=None):
    hijos = {}
    for p, pp, *_ in (filas if filas is not None else procesos()):
        hijos.setdefault(pp, []).append(p)
    todos, pila = [], [pid]
    while pila:
        for h in hijos.get(pila.pop(), []):
            todos.append(h)
            pila.append(h)
    return todos


def carga_externa():
    # nucleos que usan los procesos ajenos al barrido (suma del %CPU de ps sin el driver ni sus hijos)
    # y los 5 que mas usan; NaN si no hay ps
    filas = procesos()
    if not filas:
        return float("nan"), []
    propios = {os.getpid()} | set(descendientes(os.getpid(), filas))
    ajenos = sorted(((cpu, nom) for p, _, cpu, nom in filas if p not in propios), reverse=True)
    return sum(cpu for cpu, _ in ajenos) / 100, ajenos[:5]


def anotar(ruta, texto):
    with open(ruta, "a") as f:
        f.write(texto.rstrip("\n") + "\n")


def leer_hechas(ruta_csv):
    # (version, n_tr, n_te, p, rep, modo) que ya estan completas en tiempos.csv
    if not os.path.exists(ruta_csv):
        return set()
    hechas = set()
    with open(ruta_csv) as f:
        for r in csv.DictReader(f):
            if None in r or any(v in (None, "") for v in r.values()):
                continue                 # fila cortada (por ejemplo, si se apago la maquina)
            try:
                hechas.add((r["version"], int(r["n_tr"]), int(r["n_te"]), int(r["p"]), int(r["rep"]), r["modo"]))
            except ValueError:
                pass
    return hechas


def reparar_csv(ruta_csv):
    # si la ultima fila quedo cortada (sin salto de linea), se cierra para que no se le pegue
    # la siguiente; esa fila incompleta no cuenta como medida (leer_hechas) y analisis.py la descarta
    if os.path.exists(ruta_csv) and os.path.getsize(ruta_csv) > 0:
        with open(ruta_csv, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")


def purgar(ruta_csv, c, rep):
    # quita de tiempos.csv las filas de (c, rep) que hayan quedado de un mpirun a medias (por ejemplo,
    # solo el modo B), para que al medirla de nuevo no queden filas repetidas; devuelve cuantas quito.
    # La clave lleva el modo: el T_s de n0 (modo S) y la beta 3 con n0 y p = 1 (A y B) son otra medicion
    reparar_csv(ruta_csv)
    if not os.path.exists(ruta_csv):
        return 0
    claves = {",".join([c["version"], str(c["n_tr"]), str(c["n_te"]), str(c["p"]), str(rep), m]) + ","
              for m in c["modos"]}
    with open(ruta_csv) as f:
        lineas = f.readlines()
    quedan = lineas[:1] + [l for l in lineas[1:] if not any(l.startswith(cl) for cl in claves)]
    if len(quedan) < len(lineas):
        with open(ruta_csv + ".tmp", "w") as f:
            f.writelines(quedan)
        os.replace(ruta_csv + ".tmp", ruta_csv)
    return len(lineas) - len(quedan)


def hecha(c, rep, hechas):
    return all((c["version"], c["n_tr"], c["n_te"], c["p"], rep, m) in hechas for m in c["modos"])


def detener(proc):
    # SIGTERM a mpirun, que se lo pasa a sus procesos; si en 30 s no termino, SIGKILL a mpirun y a todos
    # sus descendientes (Open MPI deja cada proceso en su propio grupo, asi que matar el grupo no basta).
    # Devuelve (stdout, stderr, procesos que siguieran vivos)
    hijos = descendientes(proc.pid)
    proc.terminate()
    try:
        out, err = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        hijos = sorted(set(hijos) | set(descendientes(proc.pid)))
        for pid in [proc.pid] + hijos:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        out, err = proc.communicate()
    time.sleep(1)
    vivos = []
    for pid in hijos:
        try:
            os.kill(pid, 0)
            vivos.append(pid)
        except (ProcessLookupError, PermissionError):
            pass
    for pid in vivos:                    # ultimo intento: quedarian ocupando CPU en las mediciones siguientes
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return out, err, vivos


def lanzar(cmd, etiqueta, fallos, maximo, listo=None, antes=None):
    # corre un comando con tiempo maximo (s) y un reintento; devuelve su salida, o None si fallo.
    # 'listo' dice si el intento fallido igual alcanzo a escribir sus filas (por ejemplo, si se colgo al
    # terminar): en ese caso no se repite. 'antes' se llama antes de cada intento (quita filas a medias)
    for intento in (1, 2):
        if antes is not None:
            antes()
        proc = subprocess.Popen(cmd, cwd=CODIGO, env=ENTORNO, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        vivos = []
        try:
            out, err = proc.communicate(timeout=maximo)
            if proc.returncode == 0:
                return out
            motivo = f"termino con codigo {proc.returncode}"
        except subprocess.TimeoutExpired:
            out, err, vivos = detener(proc)
            motivo = f"se paso de {maximo:.0f} s"
        except KeyboardInterrupt:
            detener(proc)
            raise
        anotar(fallos, f"[{ahora()}] {etiqueta} (intento {intento}): {motivo}\n    {' '.join(cmd)}\n    "
                       + (err or "").strip()[-2000:].replace("\n", "\n    "))
        if vivos:
            anotar(fallos, f"    procesos que seguian vivos despues de matar a mpirun (se mataron): {vivos}")
        if listo is not None and listo():
            anotar(fallos, "    (sus filas ya estaban en tiempos.csv: no se repite)")
            return out
    return None


def salida(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=ENTORNO).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def energia():
    # enchufada o a bateria, modo de bajo consumo y estado termico (macOS); [] si no hay pmset
    if not shutil.which("pmset"):
        return [], []
    bat = salida(["pmset", "-g", "batt"]).splitlines()
    bajo = [l.strip() for l in salida(["pmset", "-g"]).splitlines() if "lowpowermode" in l]
    term = [l.strip() for l in salida(["pmset", "-g", "therm"]).splitlines() if l.strip()]
    lineas = [f"energia: {bat[0].strip() if bat else '?'}  {bajo[0] if bajo else ''}", "termico: " + "  ".join(term)]
    avisos = []
    if bat and "Battery Power" in bat[0]:
        avisos.append("la Mac esta a bateria: conviene enchufarla (a bateria puede bajar el rendimiento)")
    if bajo and bajo[0].split()[-1] != "0":
        avisos.append("el modo de bajo consumo esta activado: conviene apagarlo")
    return lineas, avisos


def escribir_entorno(ruta, malla, reps, rapido):
    externa, top = carga_externa()
    lineas = [f"== Inicio: {ahora()}  carga: {carga():.2f}  ({salida(['uptime'])})",
              f"carga externa: {externa:.2f} nucleos (los que mas usan: "
              + ", ".join(f"{os.path.basename(n)} {cpu:.0f}%" for cpu, n in top) + ")",
              f"comando: {' '.join([PY] + sys.argv)}",
              f"malla: {len(malla)} configuraciones, {reps} repeticiones" + (" (--rapido)" if rapido else ""),
              "hilos: " + " ".join(f"{h}=1" for h in HILOS)]
    for clave in ["machdep.cpu.brand_string", "hw.physicalcpu", "hw.logicalcpu", "hw.memsize",
                  "hw.perflevel0.name", "hw.perflevel0.physicalcpu", "hw.perflevel1.name", "hw.perflevel1.physicalcpu"]:
        valor = salida(["sysctl", "-n", clave])
        if valor:
            lineas.append(f"{clave}: {valor}")
    lineas.append(salida(["sw_vers"]).replace("\n", "  ") or platform.platform())
    lineas.append(salida([PY, "-c", "import sys, numpy, sklearn, mpi4py, matplotlib, pandas; from mpi4py import MPI; "
                          "print('Python', sys.version.split()[0], '| numpy', numpy.__version__, '| scikit-learn', "
                          "sklearn.__version__, '| mpi4py', mpi4py.__version__, '| matplotlib', matplotlib.__version__, "
                          "'| pandas', pandas.__version__); print(MPI.Get_library_version().splitlines()[0]); "
                          "print('Wtick', MPI.Wtick())"]).replace("\x00", ""))
    mpirun = salida(["mpirun", "--version"])
    lineas.append(mpirun.splitlines()[0] if mpirun else "mpirun: ?")
    en, avisos = energia()
    lineas += en
    if externa > CARGA_ALTA:
        avisos.append(f"otros programas usan {externa:.1f} nucleos (mas de {CARGA_ALTA}): conviene cerrarlos "
                      "(" + ", ".join(f"{os.path.basename(n)} {cpu:.0f}%" for cpu, n in top) + ")")
    lineas += [f"AVISO: {a}" for a in avisos]
    anotar(ruta, "\n".join(lineas))
    for a in avisos:
        print(f"AVISO: {a}", flush=True)


def paso0(carpeta, fallos):
    ruta = os.path.join(carpeta, "validacion.txt")
    if not (os.path.exists(ruta) and "Resumen:" in open(ruta).read()):
        print(f"[{ahora()}] Paso 0: validacion (unos 4 min) ...", flush=True)
        lanzar([PY, "validar.py", ruta], "validacion", fallos, 1800)
    resumen = [l for l in open(ruta) if l.startswith("Resumen:")] if os.path.exists(ruta) else []
    print("   ", resumen[0].strip() if resumen else "la validacion fallo: ver fallos.log", flush=True)
    if resumen and "revisar" in resumen[0]:
        anotar(fallos, f"[{ahora()}] validacion: {resumen[0].strip()} (ver validacion.txt)")
    ruta = os.path.join(carpeta, "pingpong.csv")
    if not os.path.exists(ruta):
        print(f"[{ahora()}] Paso 0: ping-pong ...", flush=True)
        lanzar(MPIRUN + ["2", PY, "ping_pong.py", "--csv", ruta], "ping-pong", fallos, 120)
    ruta = os.path.join(carpeta, "pruebas_extra.txt")
    if not os.path.exists(ruta):
        print(f"[{ahora()}] Paso 0: pruebas_extra.py con 1, 4 y 8 copias ...", flush=True)
        texto = [f"== Pruebas aparte {ahora()}  carga externa: {carga_externa()[0]:.2f} nucleos"]
        for p in (1, 4, 8):
            out = lanzar(MPIRUN + [str(p), PY, "pruebas_extra.py"], f"pruebas_extra p = {p}", fallos, 300)
            texto.append(out.strip() if out is not None else f"(fallo con p = {p}: ver fallos.log)")
        anotar(ruta, "\n".join(texto))


def medir_base(salida_base, rep, ruta_csv, externa):
    # el codigo base del profesor imprime su tiempo y su precision: se agregan como version 0
    t = re.search(r"Execution time \(sequential\): ([\d.]+)", salida_base).group(1)
    acc = re.search(r"Accuracy: ([\d.]+)", salida_base).group(1)
    fila = dict(version="0", n_tr=N_TR, n_te=N_TE, p=1, rep=rep, modo="S", W=0, T_ej=t, T_bcast=0,
                T_scatter=0, T_comp_max=0, T_comp_prom=0, T_gather=0, T_comp2=0, acc=acc, carga=f"{externa:.3f}",
                cpu=float("nan"), hora=time.strftime("%Y-%m-%dT%H:%M:%S"))
    nuevo = not os.path.exists(ruta_csv) or os.path.getsize(ruta_csv) == 0
    with open(ruta_csv, "a") as f:
        if nuevo:
            f.write(",".join(COLUMNAS) + "\n")
        f.write(",".join(str(fila[c]) for c in COLUMNAS) + "\n")


def minutos(s):
    return f"{s / 60:.1f} min"


parser = argparse.ArgumentParser()
parser.add_argument("carpeta", nargs="?", default="resultados_nuevos")
parser.add_argument("--reps", type=int, default=None)       # repeticiones (por defecto 4; 2 con --rapido)
parser.add_argument("--rapido", action="store_true")        # malla chica para probar el barrido
parser.add_argument("--plan", action="store_true")          # solo muestra la malla y la duracion estimada
args = parser.parse_args()
reps = args.reps if args.reps is not None else (2 if args.rapido else REPS)
carpeta = os.path.abspath(args.carpeta)
malla = armar_malla(args.rapido)
# orden: repeticiones afuera; la malla hacia adelante en las pares y hacia atras en las impares
plan = [(rep, i) for rep in range(reps) for i in (range(len(malla)) if rep % 2 == 0 else reversed(range(len(malla))))
        if rep < reps_de(malla[i], reps)]
if reps % 2 == 1:
    print(f"AVISO: con {reps} repeticiones (impar) la mediana no cancela una deriva de la maquina "
          "ni equilibra que modo va primero; conviene un numero par", flush=True)

if args.plan:
    for i, c in enumerate(malla):
        print(f"{i:>3}  {nombre(c)}  reps = {reps_de(c, reps)}  ~{estimar(c):6.1f} s por mpirun")
    total = sum(estimar(malla[i]) + PAUSA * (estimar(malla[i]) > LARGA) for _, i in plan)
    print(f"{len(malla)} configuraciones, {len(plan)} mpirun: unos {minutos(total)} de malla, mas el paso 0 (unos 4 min)")
    sys.exit(0)

os.makedirs(carpeta, exist_ok=True)
ruta_csv = os.path.join(carpeta, "tiempos.csv")
fallos = os.path.join(carpeta, "fallos.log")
ruta_malla = os.path.join(carpeta, "malla.json")
if os.path.exists(ruta_csv) and os.path.getsize(ruta_csv) > 0:
    # solo se retoma un tiempos.csv de este barrido: mismas columnas y con malla.json
    with open(ruta_csv) as f:
        cabecera = f.readline().strip().split(",")
    if cabecera != COLUMNAS or not os.path.exists(ruta_malla):
        sys.exit(f"{ruta_csv} no es de este barrido ({'otras columnas' if cabecera != COLUMNAS else 'sin malla.json'}; "
                 "por ejemplo, el de septiembre): no se toca. Use otra carpeta.")
if os.path.exists(ruta_malla):
    with open(ruta_malla) as f:
        anterior = json.load(f)
    if anterior["rapido"] != args.rapido:
        sys.exit(f"{carpeta} tiene una corrida {'con' if anterior['rapido'] else 'sin'} --rapido: use otra carpeta")
reparar_csv(ruta_csv)
hechas = leer_hechas(ruta_csv)
pendientes = [(rep, i) for rep, i in plan if not hecha(malla[i], rep, hechas)]
paso0_listo = all(os.path.exists(os.path.join(carpeta, a)) for a in ("validacion.txt", "pingpong.csv", "pruebas_extra.txt"))
if not pendientes and paso0_listo:
    print(f"{carpeta} ya tiene una corrida completa ({len(plan)} mediciones en tiempos.csv): no se pisa. "
          "Use otra carpeta para volver a medir.")
    sys.exit(0)
reanuda = len(hechas) > 0 or os.path.exists(os.path.join(carpeta, "entorno.txt"))
with open(ruta_malla, "w") as f:
    json.dump(dict(rapido=args.rapido, reps=reps, configuraciones=len(malla)), f)
if shutil.which("caffeinate"):           # que la Mac no se duerma mientras corre el barrido (termina con el driver)
    subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"{'Retoma' if reanuda else 'Empieza'} el barrido en {carpeta}: {len(pendientes)} de {len(plan)} mpirun pendientes",
      flush=True)
escribir_entorno(os.path.join(carpeta, "entorno.txt"), malla, reps, args.rapido)
if reanuda:
    anotar(os.path.join(carpeta, "entorno.txt"), f"(retoma: {len(plan) - len(pendientes)} de {len(plan)} ya medidas)")

inicio = time.time()
paso0(carpeta, fallos)
print(f"[{ahora()}] Paso 0 listo en {minutos(time.time() - inicio)}", flush=True)

inicio_malla = time.time()
duracion = {}                            # duracion medida de cada configuracion (ultima repeticion)
real = estimado = 0.0                    # para corregir el estimado a priori con lo ya medido
n_fallos = 0
for j, (rep, i) in enumerate(pendientes, 1):
    c = malla[i]
    if any((c["version"], c["n_tr"], c["n_te"], c["p"], rep, m) in hechas for m in c["modos"]):
        n = purgar(ruta_csv, c, rep)     # quedo a medias: se vuelve a medir entera
        anotar(fallos, f"[{ahora()}] rep {rep}  {nombre(c)}: estaba a medias; se quitaron sus {n} filas y se mide de nuevo")
    externa, _ = carga_externa()
    t0 = time.time()
    out = lanzar(comando(c, rep, ruta_csv, externa), f"rep {rep}  {nombre(c)}", fallos, tiempo_max(c),
                 listo=lambda: hecha(c, rep, leer_hechas(ruta_csv)), antes=lambda: purgar(ruta_csv, c, rep))
    if out is not None and c["version"] == "0":
        try:
            medir_base(out, rep, ruta_csv, externa)
        except AttributeError:           # no encontro el tiempo o la precision en la salida
            anotar(fallos, f"[{ahora()}] rep {rep}  {nombre(c)}: salida sin tiempo\n    {out[-500:]}")
            out = None
    dt = time.time() - t0
    n_fallos += out is None
    duracion[i] = dt
    real += dt
    estimado += estimar(c)
    factor = real / estimado
    falta = sum(duracion.get(ii, estimar(malla[ii]) * factor) + PAUSA * (estimar(malla[ii]) > LARGA)
                for _, ii in pendientes[j:])
    print(f"[{j:>3}/{len(pendientes)}] rep {rep}  {nombre(c)}  {dt:6.1f} s  carga ext. {externa:4.2f}"
          f"{'  FALLO' if out is None else ''}  | {minutos(time.time() - inicio_malla)}  | faltan ~{minutos(falta)}"
          f" (fin ~{time.strftime('%H:%M', time.localtime(time.time() + falta))})", flush=True)
    if dt > LARGA and j < len(pendientes):
        time.sleep(PAUSA)                # pausa despues de una corrida larga (temperatura, procesos que terminan)

total = time.time() - inicio
en, avisos = energia()
anotar(os.path.join(carpeta, "entorno.txt"), "\n".join([f"== Fin: {ahora()}  carga: {carga():.2f}  carga externa: "
                                                        f"{carga_externa()[0]:.2f}  duracion: {minutos(total)}  fallos: {n_fallos}"]
                                                       + en + [f"AVISO: {a}" for a in avisos]))
print(f"Listo en {minutos(total)} ({n_fallos} fallos{', ver fallos.log' if n_fallos else ''}). "
      f"Siguiente paso: python analisis.py --resultados {args.carpeta}", flush=True)
