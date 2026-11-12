"""
Proyecto CS4052 - Caso 5 KNN - Grupo 7
Analisis de resultados: lee DIR/tiempos.csv (y DIR/pingpong.csv) y genera
    - DIR/figuras/*.pdf      (graficas del informe; las mismas en DIR/figuras/png/, a 200 dpi)
    - DIR/tablas/*.tex       (tablas del informe)
    - DIR/resumen.json       (numeros clave)
Cada configuracion (version, n_tr, n_te, p, modo) se resume con la MEDIANA de sus
repeticiones. T_ej sale del modo A y las fases del modo B. Las series se arman por tamano
(n_tr, n_te), no por etiqueta: una configuracion que esta en varias series (n0 con p = 1
esta en la fuerte, la debil, la iso y la de control) se mide una sola vez y todas comparten
esa linea base. Si falta una serie o un punto, se avisa y se sigue.

Ejecucion:  python analisis.py [--resultados resultados]
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--resultados", default="resultados")
args = parser.parse_args()
DIR = args.resultados
CSV = f"{DIR}/tiempos.csv"
FIG = f"{DIR}/figuras"
TAB = f"{DIR}/tablas"
os.makedirs(f"{FIG}/png", exist_ok=True)
os.makedirs(TAB, exist_ok=True)

d, k = 64, 3
FLOP_POR_PAR = 3 * d                     # d restas + d productos + (d-1) sumas + 1 raiz
P = [1, 2, 4, 6, 8]
FACS = [0.25, 1, 4, 16]
NOMBRE = {0.25: "n/4", 1: "n0", 4: "4n", 16: "16n"}
TNOM = {0.25: "$n/4$", 1: "$n_0$", 4: "$4n$", 16: "$16n$"}
N_TR0, N_TE0 = 1437, 360                 # split original del codigo base (n0)


def tamano(f, n):
    # mismo tamano que replicar() de knn_mpi_beta3.py
    return int(f * n) if f < 1 else int(round(f * n))


# series por tamano (n_tr, n_te)
TAM = {fac: (tamano(fac, N_TR0), tamano(fac, N_TE0)) for fac in FACS}                       # fuerte: n fijo
DEBIL = {p: (tamano(np.sqrt(p), N_TR0), tamano(np.sqrt(p), N_TE0)) for p in [1, 2, 4, 8]}    # n ~ raiz(p)
ISO = {p: (tamano(p ** 2, N_TR0), tamano(p ** 2, N_TE0)) for p in [1, 2, 4, 8]}              # n ~ k p^2
CONTROL = {p: (N_TR0 * p, N_TE0) for p in P}                                                  # n_te fijo


def miles(x):
    return f"{int(x):,}".replace(",", "\\,")


COLOR = {0.25: "#1b9e77", 1: "#d95f02", 4: "#7570b3", 16: "#e7298a"}

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "legend.fontsize": 8, "figure.dpi": 150})
AVISOS = []


def aviso(texto):
    # series o puntos que faltan: se avisa y se sigue (las figuras y tablas quedan con huecos)
    AVISOS.append(texto)
    print("AVISO:", texto)


def fmt(x, dec=3):
    return "--" if not np.isfinite(x) else f"{x:.{dec}f}".replace(".", ",")


df = pd.read_csv(CSV, on_bad_lines="warn")
PROTOCOLO_NUEVO = {"W", "hora"} <= set(df.columns)
if not PROTOCOLO_NUEVO:
    aviso(f"{CSV} no tiene las columnas W y hora: son datos del protocolo viejo (varias repeticiones seguidas "
          "en un mismo mpirun, y la misma configuracion medida en varias series), no del barrido nuevo")
for col in ["carga", "cpu"]:
    if col not in df.columns:
        df[col] = np.nan
filas = len(df)
df = df[df.drop(columns=["carga", "cpu"]).notna().all(axis=1)]                    # filas cortadas
if len(df) < filas:
    aviso(f"se descartaron {filas - len(df)} filas incompletas de {CSV}")
df["version"] = df["version"].astype(str).str.replace(r"\.0$", "", regex=True)   # 0, 1, 2, 3 y 3b
CLAVE_FILA = ["version", "n_tr", "n_te", "p", "rep", "modo"]
repetidas = df.duplicated(CLAVE_FILA, keep="last") & PROTOCOLO_NUEVO
if repetidas.any():                      # por ejemplo, una medicion retomada a mano: queda la ultima
    aviso(f"{int(repetidas.sum())} filas repetidas (misma version, tamano, p, repeticion y modo): se usa la ultima")
    df = df[~repetidas]
# maquina libre? cpu < 0,9: algun proceso fue desalojado en la corrida medida; carga: otros programas
desalojadas, con_cpu = int((df.cpu < 0.9).sum()), int(df.cpu.notna().sum())
if con_cpu and desalojadas > 0.05 * con_cpu:
    aviso(f"en {desalojadas} de {con_cpu} corridas medidas algun proceso tuvo menos del 90 % de la CPU: "
          "la maquina no estaba libre")
if PROTOCOLO_NUEVO and df.carga.median() > 1.5:
    aviso(f"carga externa mediana de {df.carga.median():.1f} nucleos (otros programas): la maquina no estaba libre")
# pasos 5 a 7 (gather o reduce, mezcla, votacion y precision) = tf - t4, un solo intervalo de cada corrida
df["T_57"] = df["T_gather"] + df["T_comp2"]


def barras(x):
    # barras de error de una configuracion: el rango intercuartil (percentiles 25 y 75) si hay
    # 5 o mas repeticiones (--reps 6); con menos (las 4 por defecto, las 2 de 115 008 imagenes o
    # la prueba --rapido), el minimo y el maximo, porque los cuartiles de 2 a 4 datos no dicen mucho
    x = np.asarray(x, dtype=float)
    if len(x) >= 5:
        return tuple(np.percentile(x, [25, 75]))
    return x.min(), x.max()


# ---------------------------------------------------------------------------
# 0. Mediana de las repeticiones por (version, n_tr, n_te, p, modo)
#    modo A: T_ej (sin barreras); modo B: fases (con barreras); modo S: T_s
# ---------------------------------------------------------------------------
CLAVE = ["version", "n_tr", "n_te", "p", "modo"]
grupos = df.groupby(CLAVE)
MED = grupos[["T_ej", "T_bcast", "T_scatter", "T_comp_max", "T_comp_prom", "T_gather", "T_comp2",
              "T_57", "acc", "carga", "cpu"]].median()
MED["reps"] = grupos.size()
# T_comm y T_comp son sumas de fases que no son contiguas: se suman las medianas de cada fase, asi
# T_comm = T_bcast + T_scatter + T_gather y T_comp = T_comp_max + T_comp2 tambien en las tablas
MED["T_comm"] = MED.T_bcast + MED.T_scatter + MED.T_gather
MED["T_comp"] = MED.T_comp_max + MED.T_comp2
for col in ["T_ej", "T_57"]:
    lh = grupos[col].apply(barras)
    MED[col + "_lo"] = lh.map(lambda b: b[0])
    MED[col + "_hi"] = lh.map(lambda b: b[1])


def med(version, n_tr, n_te, p, modo):
    # medianas de una configuracion (None si no se midio)
    clave = (str(version), int(n_tr), int(n_te), int(p), modo)
    return MED.loc[clave] if clave in MED.index else None


# ---------------------------------------------------------------------------
# 1. Resumen por (fac, p) de la beta 3 (y de la 3b), escalabilidad fuerte
# ---------------------------------------------------------------------------
def serie_fuerte(version):
    filas = []
    for fac in FACS:
        n_tr, n_te = TAM[fac]
        S = med("3", n_tr, n_te, 1, "S")            # T_s: version secuencial del mismo calculo
        for p in P:
            A, B = med(version, n_tr, n_te, p, "A"), med(version, n_tr, n_te, p, "B")
            if A is None:
                continue
            if B is None:
                aviso(f"beta {version}, {NOMBRE[fac]}, p = {p}: falta el modo B (fases)")
                B = pd.Series(np.nan, index=MED.columns)
            filas.append(dict(fac=fac, p=p, n_tr=n_tr, n_te=n_te, Ts=S.T_ej if S is not None else np.nan,
                              T=A.T_ej, T_lo=A.T_ej_lo, T_hi=A.T_ej_hi, reps=int(A.reps),
                              T_bcast=B.T_bcast, T_scatter=B.T_scatter, T_gather=B.T_gather,
                              T_comp1=B.T_comp_max, T_comp1_prom=B.T_comp_prom, T_comp2=B.T_comp2,
                              T_comp=B.T_comp, T_comm=B.T_comm, T_57=B.T_57, T_57_lo=B.T_57_lo,
                              T_57_hi=B.T_57_hi, acc=A.acc, carga=A.carga))
    return pd.DataFrame(filas, columns=["fac", "p", "n_tr", "n_te", "Ts", "T", "T_lo", "T_hi", "reps",
                                        "T_bcast", "T_scatter", "T_gather", "T_comp1", "T_comp1_prom",
                                        "T_comp2", "T_comp", "T_comm", "T_57", "T_57_lo", "T_57_hi",
                                        "acc", "carga"])


R = serie_fuerte("3")
R3b = serie_fuerte("3b")
for fac in FACS:
    if not (R.fac == fac).any():
        aviso(f"falta la serie fuerte {NOMBRE[fac]} de la beta 3 (n_tr = {TAM[fac][0]}, n_te = {TAM[fac][1]})")
    elif R[R.fac == fac].Ts.isna().all():
        aviso(f"falta T_s (modo S) de {NOMBRE[fac]}: sin speedup ni eficiencia en ese tamano")
    if not (R3b.fac == fac).any():
        aviso(f"falta la serie fuerte {NOMBRE[fac]} de la beta 3b")
if len(R) == 0:
    raise SystemExit(f"No hay mediciones de la beta 3 con n fijo en {CSV}: no hay nada que analizar.")
FAC_OK = [fac for fac in FACS if (R.fac == fac).any()]          # tamanos medidos
for RR in (R, R3b):
    RR["S"] = RR.Ts / RR["T"]
    RR["E"] = RR.S / RR.p
    RR["T_err"] = (RR.T_hi - RR.T_lo) / 2     # tablas: mediana +- la mitad del rango de barras()
R["FLOP"] = FLOP_POR_PAR * R.n_tr * R.n_te
R["GFLOPs"] = R.FLOP / R["T"] / 1e9
R["G"] = R.T_comp / R.T_comm
R["desb"] = R.T_comp1 / R.T_comp1_prom
R["dTdp"] = np.nan
for fac in FAC_OK:
    r = R[R.fac == fac].sort_values("p")
    R.loc[r.index[1:], "dTdp"] = np.diff(r["T"].values) / np.diff(r.p.values)

# ---------------------------------------------------------------------------
# 2. Normalizacion: constantes de cada termino del modelo teorico
#    T_p = c1*n_te*n_tr/p + c2*n_te*k*p + log2(p)(a1 + n_te*b1) + (log2(p)*a2 + n_tr*b2) + (log2(p)*a3 + p*n_te*k*b3)
#    scatter y gather en arbol, como en MPI: O(a log p + b m) (U2.4 y correo del profesor del 6-nov)
# ---------------------------------------------------------------------------
R["pares_q"] = R.n_te * np.ceil(R.n_tr / R.p)
R["n_q"] = np.ceil(R.n_tr / R.p)
R["c1"] = R.T_comp1 / R.pares_q                              # s por par prueba-entrenamiento
R["c2"] = R.T_comp2 / (R.n_te * k * R.p)                     # s por candidato en el maestro
c2 = np.nanmedian(R.c2)


def ajuste(X, y, nombre):
    # minimos cuadrados con error relativo (pesos 1/y); si una constante sale
    # negativa (sin sentido fisico) se fija en 0 y se vuelven a ajustar las demas
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(y) & (y > 0) & np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    sol = np.zeros(X.shape[1])
    if len(y) < X.shape[1]:
        aviso(f"muy pocos puntos para ajustar {nombre} ({len(y)}): sus constantes quedan en 0")
        return sol
    w = 1.0 / y
    libres = list(range(X.shape[1]))
    while libres:
        c, *_ = np.linalg.lstsq(X[:, libres] * w[:, None], y * w, rcond=None)
        if np.all(c >= 0):
            sol[libres] = c
            break
        libres = [j for j, cj in zip(libres, c) if cj > 0]
    return sol


# T_comp1 = n_te*(n_tr/p)*(cd + cs*log2(n_tr/p)): distancias (d) + ordenamiento (log), como en el parcial
cd, cs = ajuste(np.column_stack([R.pares_q, R.pares_q * np.log2(R.n_q)]), R.T_comp1, "T_comp1")
c1 = np.nanmedian(R[R.fac >= 1].c1)                         # costo medio por par (referencia)

m = R[R.p >= 2]
a1, b1 = ajuste(np.column_stack([np.log2(m.p), np.log2(m.p) * m.n_te]), m.T_bcast, "bcast")
a2, b2 = ajuste(np.column_stack([np.log2(m.p), m.n_tr]), m.T_scatter, "scatter")
a3, b3 = ajuste(np.column_stack([np.log2(m.p), m.p * m.n_te * k]), m.T_gather, "gather")
CONST = dict(cd=cd, cs=cs, c1=c1, c2=c2, a1=a1, b1=b1, a2=a2, b2=b2, a3=a3, b3=b3)


def T_teo(n_tr, n_te, p, parte="total"):
    p = np.asarray(p, dtype=float)
    comp1 = n_te * (n_tr / p) * (cd + cs * np.log2(np.maximum(n_tr / p, 2)))
    comp = comp1 + c2 * n_te * k * p
    bc = np.log2(p) * (a1 + n_te * b1)
    sc = np.log2(p) * a2 + n_tr * b2
    ga = np.log2(p) * a3 + p * n_te * k * b3
    partes = dict(comp=comp, comm=bc + sc + ga, bcast=bc, scatter=sc, gather=ga,
                  uno_p=comp1, lin_p=c2 * n_te * k * p + p * n_te * k * b3 + n_tr * b2, log_p=bc + np.log2(p) * (a2 + a3))
    return comp + bc + sc + ga if parte == "total" else partes[parte]


R["T_teo"] = T_teo(R.n_tr, R.n_te, R.p)
R["err_rel"] = (R["T_teo"] - R["T"]).abs() / R["T"]
y, yt = np.log(R["T"]), np.log(R["T_teo"])
R2_log = 1 - np.sum((y - yt) ** 2) / np.sum((y - y.mean()) ** 2)
R2 = 1 - np.sum((R["T"] - R["T_teo"]) ** 2) / np.sum((R["T"] - R["T"].mean()) ** 2)

# ---------------------------------------------------------------------------
# 3. p optimo: T(p) = A/p + B p + C log2 p + D  ->  T'(p) = 0
# ---------------------------------------------------------------------------
popt = []
for fac in FAC_OK:
    r = R[R.fac == fac].iloc[0]
    ps = np.linspace(1, 4096, 400000)
    p_num = ps[np.argmin(T_teo(r.n_tr, r.n_te, ps))]
    A_ = r.n_te * r.n_tr * (cd + cs * np.log2(r.n_tr / p_num))   # costo por par en p_opt
    B_ = c2 * r.n_te * k + r.n_te * k * b3
    C_ = a1 + a2 + a3 + r.n_te * b1
    ln2 = np.log(2)
    p_an = (-C_ / ln2 + np.sqrt((C_ / ln2) ** 2 + 4 * A_ * B_)) / (2 * B_) if B_ > 0 else np.nan
    rr = R[R.fac == fac]
    p_exp = int(rr.loc[rr["T"].idxmin(), "p"])
    # nivel examen: a = b = t_c = 1 (sin constantes)
    A1, B1, C1 = r.n_te * r.n_tr, r.n_te * k * 2, 3 + r.n_te
    p_ex = (-C1 / ln2 + np.sqrt((C1 / ln2) ** 2 + 4 * A1 * B1)) / (2 * B1)
    popt.append(dict(fac=fac, n_tr=int(r.n_tr), n_te=int(r.n_te), A=A_, B=B_, C=C_,
                     p_examen=p_ex, p_teo=p_an, p_num=p_num, p_exp=p_exp))
POPT = pd.DataFrame(popt, columns=["fac", "n_tr", "n_te", "A", "B", "C", "p_examen", "p_teo", "p_num", "p_exp"])

# ---------------------------------------------------------------------------
# 4. Escalabilidad debil (Gustafson, n ~ raiz(p)), isoeficiencia (n ~ k p^2) y control
#    (n_te fijo, n_tr ~ p). Se buscan por tamano; con p = 1 todas usan n0 con p = 1.
# ---------------------------------------------------------------------------
esc = []
for ley, serie in [("debil_n", DEBIL), ("iso_n", ISO)]:
    for p, (n_tr, n_te) in serie.items():
        A, S = med("3", n_tr, n_te, p, "A"), med("3", n_tr, n_te, 1, "S")
        if A is None:
            aviso(f"falta el punto p = {p} de la serie {'debil' if ley == 'debil_n' else 'iso'} "
                  f"(n_tr = {n_tr}, n_te = {n_te})")
            continue
        if S is None:
            aviso(f"falta T_s de n_tr = {n_tr}, n_te = {n_te}: sin E en ese punto")
        Ts = S.T_ej if S is not None else np.nan
        esc.append(dict(ley=ley, p=p, n_tr=n_tr, n_te=n_te, Ts=Ts, T=A.T_ej, T_lo=A.T_ej_lo, T_hi=A.T_ej_hi,
                        reps=int(A.reps), E=Ts / (p * A.T_ej), acc=A.acc))
ESC = pd.DataFrame(esc, columns=["ley", "p", "n_tr", "n_te", "Ts", "T", "T_lo", "T_hi", "reps", "E", "acc"])
ESC["T_teo"] = [float(T_teo(f.n_tr, f.n_te, f.p)) for _, f in ESC.iterrows()]
ESC["Ew_teo"] = np.nan
ESC["Ew"] = np.nan
e = ESC[ESC.ley == "debil_n"]
if len(e) and (e.p == 1).any():
    # eficiencia debil E_w = T(1)/T(p) (trabajo por proceso constante), con T(1) = n0 con p = 1
    ESC.loc[e.index, "Ew_teo"] = e[e.p == 1]["T_teo"].values[0] / e["T_teo"]
    ESC.loc[e.index, "Ew"] = e[e.p == 1]["T"].values[0] / e["T"]
elif len(e):
    aviso("falta T(1) de la serie debil (n0 con p = 1): sin E_w")

ctl = []
for p, (n_tr, n_te) in CONTROL.items():
    A, B = med("3", n_tr, n_te, p, "A"), med("3", n_tr, n_te, p, "B")
    if A is None:
        aviso(f"falta el punto p = {p} de la serie de control (n_tr = {n_tr}, n_te = {n_te})")
        continue
    ctl.append(dict(p=p, n_tr=n_tr, n_te=n_te, T=A.T_ej, T_lo=A.T_ej_lo, T_hi=A.T_ej_hi,
                    T_comp1=B.T_comp_max if B is not None else np.nan))
CTL = pd.DataFrame(ctl, columns=["p", "n_tr", "n_te", "T", "T_lo", "T_hi", "T_comp1"])
CTL["Ew"] = CTL[CTL.p == 1]["T"].values[0] / CTL["T"] if (CTL.p == 1).any() else np.nan

# ---------------------------------------------------------------------------
# 5. Versiones beta (datos originales) y codigo base del profesor
# ---------------------------------------------------------------------------
betas = []
for v in [1, 2, 3]:
    for p in P:
        A = med(v, N_TR0, N_TE0, p, "A")
        if A is not None:
            betas.append(dict(version=v, p=p, T=A.T_ej, T_lo=A.T_ej_lo, T_hi=A.T_ej_hi, acc=A.acc))
    if not any(b["version"] == v for b in betas):
        aviso(f"falta la beta {v} con n0")
BET = pd.DataFrame(betas, columns=["version", "p", "T", "T_lo", "T_hi", "acc"])
B0 = med(0, N_TR0, N_TE0, 1, "S")
T_base = B0.T_ej if B0 is not None else np.nan
if B0 is None:
    aviso("falta el codigo base del profesor (version 0): F8 queda sin su linea")

# ---------------------------------------------------------------------------
# 6. Precision de la serie fuerte (la misma para todo p y para las betas 3 y 3b)
# ---------------------------------------------------------------------------
prec = []
for fac in FAC_OK:
    n_tr, n_te = TAM[fac]
    a = df[df.version.isin(["3", "3b"]) & (df.n_tr == n_tr) & (df.n_te == n_te)].acc
    prec.append(dict(fac=fac, n=n_tr + n_te, n_tr=n_tr, n_te=n_te, acc=a.median(), acc_min=a.min(),
                     acc_max=a.max(), corridas=len(a)))
    if a.max() - a.min() > 1e-12:
        aviso(f"la precision de {NOMBRE[fac]} cambia entre corridas ({a.min():.4f} a {a.max():.4f})")
PREC = pd.DataFrame(prec, columns=["fac", "n", "n_tr", "n_te", "acc", "acc_min", "acc_max", "corridas"])

# ---------------------------------------------------------------------------
# 7. Ping-pong: T(m) = alfa + m*beta, con m datos float64. Una sola recta no describe los 6 ordenes
#    de magnitud de m (con pickle, el costo por dato cambia con el tamano del mensaje), asi que:
#    alfa = latencia, la mediana del tiempo de ida con m <= 100; beta = pendiente por minimos
#    cuadrados (sin pesos) con m >= 1e4, el rango de las colectivas de las betas. Se guarda el error
#    relativo maximo del modelo y, como referencia, la recta ajustada con todos los m (con error relativo).
# ---------------------------------------------------------------------------
PP = {}
M_ALFA, M_BETA = 100, 1e4
if os.path.exists(f"{DIR}/pingpong.csv"):
    pp_df = pd.read_csv(f"{DIR}/pingpong.csv")
    for var, g in pp_df.groupby("variante"):
        m_, t_ = g.m.values.astype(float), g.t_ida.values.astype(float)
        chicos, grandes = m_ <= M_ALFA, m_ >= M_BETA
        if chicos.sum() < 1 or grandes.sum() < 2:
            aviso(f"ping-pong ({var}): faltan mensajes chicos (m <= {M_ALFA}) o grandes (m >= {M_BETA:.0e})")
            continue
        alfa = float(np.median(t_[chicos]))
        beta = float(np.polyfit(m_[grandes], t_[grandes], 1)[0])
        err = np.abs(alfa + beta * m_ - t_) / t_
        a1_pp, b1_pp = ajuste(np.column_stack([np.ones(len(g)), m_]), t_, f"ping-pong ({var})")
        PP[var] = dict(alfa=alfa, beta=beta, GBs=8 / beta / 1e9 if beta > 0 else np.nan,
                       err_rel_max=float(err.max()), m_err_max=int(m_[np.argmax(err)]),
                       err_rel_max_m_grandes=float(err[grandes].max()),
                       criterio=f"alfa: mediana con m <= {M_ALFA}; beta: minimos cuadrados con m >= {M_BETA:.0e}",
                       una_recta=dict(alfa=float(a1_pp), beta=float(b1_pp),
                                      err_rel_max=float((np.abs(a1_pp + b1_pp * m_ - t_) / t_).max())),
                       tamanos=len(g), m_max=int(m_.max()))
else:
    aviso(f"falta {DIR}/pingpong.csv: sin alfa ni beta")


# ===========================================================================
# FIGURAS
# ===========================================================================
def guardar(fig, nombre):
    # cada figura va en PDF (informe) y en PNG a 200 dpi con fondo blanco (presentacion)
    fig.savefig(f"{FIG}/{nombre}.pdf")
    fig.savefig(f"{FIG}/png/{nombre}.png", dpi=200, facecolor="white")
    plt.close(fig)


def barra(r, col="T"):
    # barras asimetricas alrededor de la mediana (ver barras())
    return np.vstack([r[col] - r[col + "_lo"], r[col + "_hi"] - r[col]])


def sin_datos(ax, titulo):
    ax.text(0.5, 0.5, "sin datos", transform=ax.transAxes, ha="center", va="center", color="0.5")
    ax.set_title(titulo, fontsize=9)


def leyenda(axs):
    # entradas de la leyenda comun: las del primer panel que tenga datos
    for ax in np.ravel(axs):
        h, l = ax.get_legend_handles_labels()
        if h:
            return h, l
    return [], []


def eje_p(ax, pmax=8, log=True):
    if not np.all(np.isfinite(ax.dataLim.intervalx)):
        # sin ningun punto en x (por ejemplo, todas las E en NaN porque falta el modo S): limites
        # fijos, porque la escala log no tiene de donde sacarlos y matplotlib se cae
        ax.set_xlim(0.85, pmax * 1.15)
    if log:
        ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 6, 8] if pmax == 8 else [1, 2, 4, 8, 16, 32, 64])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("p (#procesos)")


pp = np.linspace(1, 8, 200)

# F1: tiempos de ejecucion, computo y comunicacion
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))
for ax, fac in zip(axs.flat, FACS):
    r = R[R.fac == fac]
    if len(r) == 0:
        sin_datos(ax, NOMBRE[fac])
        continue
    n_tr, n_te = int(r.n_tr.iloc[0]), int(r.n_te.iloc[0])
    ax.errorbar(r.p, r["T"], yerr=barra(r), fmt="o", color="k", ms=4, capsize=2, label="$T_{ej}$ (medido)")
    ax.plot(r.p, r.T_comp, "^", color="#c2452a", ms=4, label="$T_{comp}$ (medido)")
    ax.plot(r.p, r.T_comm, "s", color="#2563c9", ms=4, label="$T_{comm}$ (medido)")
    ax.plot(pp, T_teo(n_tr, n_te, pp), "-", color="k", lw=1, label="$T_{ej}$ teórico")
    ax.plot(pp, T_teo(n_tr, n_te, pp, "comp"), "-", color="#c2452a", lw=1, label="$T_{comp}$ teórico")
    ax.plot(pp, T_teo(n_tr, n_te, pp, "comm"), "-", color="#2563c9", lw=1, label="$T_{comm}$ teórico")
    ax.set_yscale("log")
    eje_p(ax)
    ax.set_ylabel("Tiempo (s)")
    ax.set_title(f"{NOMBRE[fac]}: n = {n_tr + n_te} ($n_{{tr}}$ = {n_tr}, $n_{{te}}$ = {n_te})", fontsize=9)
h, l = leyenda(axs)
fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1))
guardar(fig, "F1_tiempos")

# F2: tiempo de cada colectiva
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, fac in zip(axs, [1, 16]):
    r = R[(R.fac == fac) & (R.p >= 2)]
    if len(r) == 0:
        sin_datos(ax, NOMBRE[fac])
        continue
    n_tr, n_te = int(r.n_tr.iloc[0]), int(r.n_te.iloc[0])
    p2 = np.linspace(2, 8, 100)
    for col, parte, c, mk, nom in [("T_bcast", "bcast", "#1b9e77", "o", "bcast"),
                                   ("T_scatter", "scatter", "#d95f02", "s", "scatter"),
                                   ("T_gather", "gather", "#7570b3", "^", "gather")]:
        ax.plot(r.p, r[col], mk, color=c, ms=4, label=f"{nom} (medido)")
        ax.plot(p2, T_teo(n_tr, n_te, p2, parte), "-", color=c, lw=1)
    ax.set_yscale("log")
    eje_p(ax)
    ax.set_xticks([2, 4, 6, 8])
    ax.set_ylabel("Tiempo (s)")
    ax.set_title(f"{NOMBRE[fac]} (n = {n_tr + n_te}); líneas: modelo ajustado", fontsize=9)
for ax in axs:                           # leyenda en el primer panel con datos
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
        break
fig.tight_layout()
guardar(fig, "F2_colectivas")

# F3: teoria normalizada vs experimento y p optimo (estilo N-cuerpos)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
pl = np.logspace(0, np.log2(512), 300, base=2)
for ax, fac in zip(axs, [1, 16]):
    r = R[R.fac == fac]
    if len(r) == 0:
        sin_datos(ax, NOMBRE[fac])
        continue
    n_tr, n_te = int(r.n_tr.iloc[0]), int(r.n_te.iloc[0])
    po = POPT[POPT.fac == fac].iloc[0]
    ax.axvspan(8, pl[-1], color="0.9", zorder=0)
    ax.text(11, 0.72, "extrapolación\n(p > 8)", transform=ax.get_xaxis_transform(), fontsize=7, color="0.4",
            zorder=1)
    ax.plot(pl, T_teo(n_tr, n_te, pl, "uno_p"), "--", color="#c2452a", lw=1, label="~1/p (cómputo)")
    ax.plot(pl, T_teo(n_tr, n_te, pl, "lin_p"), "--", color="#2563c9", lw=1, label="~p + cte (mezcla, datos de scatter y gather)")
    pl2 = pl[pl >= 2]
    ax.plot(pl2, T_teo(n_tr, n_te, pl2, "log_p"), ":", color="#1b9e77", lw=1, label="~log p (latencias y bcast)")
    ax.plot(pl, T_teo(n_tr, n_te, pl), "-", color="k", lw=1.3, label="aprox. (total)")
    ax.plot(r.p, r["T"], "+", color="red", ms=8, mew=1.5, label="experimento")
    ax.axvline(po.p_num, color="k", ls=":", lw=1)
    ax.text(po.p_num * 1.08, 0.9, f"$p_{{opt}}$ ≈ {po.p_num:.0f}", transform=ax.get_xaxis_transform(), fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("p (#procesos)")
    ax.set_ylabel("$T_{ej}$ (s)")
    ax.set_title(f"{NOMBRE[fac]} (n = {n_tr + n_te})", fontsize=9)
h, l = leyenda(axs)
fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0.15, 1, 1))
guardar(fig, "F3_popt")

# F4: speedup
fig, ax = plt.subplots(figsize=(3.5, 3.0))
for fac in FAC_OK:
    r = R[R.fac == fac]
    ax.plot(r.p, r.S, "o-", color=COLOR[fac], ms=4, lw=1, label=f"{NOMBRE[fac]} (n = {int(r.n_tr.iloc[0] + r.n_te.iloc[0])})")
ax.plot([1, 8], [1, 8], "k--", lw=1, label="S = p")
ax.set_xticks(P)
ax.set_xlabel("p (#procesos)")
ax.set_ylabel("Speedup $S = T_s/T_p$")
ax.legend(fontsize=7)
fig.tight_layout()
guardar(fig, "F4_speedup")

# F5: FLOP/s
fig, ax = plt.subplots(figsize=(3.5, 3.0))
for fac in FAC_OK:
    r = R[R.fac == fac]
    n_tr, n_te = int(r.n_tr.iloc[0]), int(r.n_te.iloc[0])
    ax.plot(r.p, r.GFLOPs, "o", color=COLOR[fac], ms=4, label=f"{NOMBRE[fac]}")
    ax.plot(pp, FLOP_POR_PAR * n_tr * n_te / T_teo(n_tr, n_te, pp) / 1e9, "-", color=COLOR[fac], lw=0.8)
ref = R[R.p == 1].sort_values("fac")                       # referencia: 16n con p = 1 (o el mayor medido)
if len(ref):
    g1 = float(ref.GFLOPs.iloc[-1])
    extra = "" if ref.fac.iloc[-1] == 16 else f" ({NOMBRE[ref.fac.iloc[-1]]})"
    ax.plot(pp, g1 * pp, "k--", lw=1, label=f"{fmt(g1, 1)} GFLOP/s · p{extra}")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
eje_p(ax)
ax.set_ylabel("FLOP/s (GFLOP/s)")
ax.legend(fontsize=7)
fig.tight_layout()
guardar(fig, "F5_flops")

# F6: eficiencia con n variable
fig, ax = plt.subplots(figsize=(3.5, 3.0))
for fac in FAC_OK:
    r = R[R.fac == fac]
    ax.plot(r.p, r.E, "o-", color=COLOR[fac], ms=4, lw=1, label=f"{NOMBRE[fac]}")
ax.axhline(1, color="k", ls="--", lw=1, label="E = 1")
ax.set_ylim(0, 1.3)
eje_p(ax)
ax.set_ylabel("Eficiencia E")
ax.legend(fontsize=7, loc="lower left")
fig.tight_layout()
guardar(fig, "F6_eficiencia")

# F7: escalabilidad fuerte, debil (Gustafson) e isoeficiencia
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax = axs[0]
for ley, c, nom in [("debil_n", "#d95f02", "débil: n ∝ √p"), ("iso_n", "#7570b3", "isoeficiencia: n ∝ p²")]:
    e = ESC[ESC.ley == ley]
    if len(e) == 0:
        continue
    ax.errorbar(e.p, e["T"], yerr=barra(e), fmt="o-", color=c, ms=4, lw=1, capsize=2, label=nom)
    for _, f in e.iterrows():
        val = f.Ew if ley == "debil_n" else f.E
        if not np.isfinite(val):
            continue
        dy = -11 if ley == "debil_n" else 5
        dx = -26 if f.p == 8 else 4    # el ultimo punto lleva la etiqueta a la izquierda
        ax.annotate(f"{100 * val:.0f}%", (f.p, f["T"]), textcoords="offset points", xytext=(dx, dy), fontsize=7, color=c)
ax.set_xticks([1, 2, 4, 8])
ax.set_xlim(0.4, 9.2)   # margen para que no se corten las etiquetas de %
ax.set_xlabel("p (#procesos)")
ax.set_ylabel("$T_{ej}$ (s)")
ax.set_yscale("log")
ax.set_ylim(0.015, 60)
ax.set_title("n crece con p (débil: $E_w=T(1)/T(p)$; iso: $E$)", fontsize=8)
ax.legend(fontsize=7)
ax = axs[1]
r = R[R.fac == 1]
ax.plot(r.p, r.E, "o-", color="#1b9e77", ms=4, lw=1, label="fuerte (n0 fijo)")
for ley, c, nom in [("debil_n", "#d95f02", "débil: n ∝ $\\sqrt{p}$"), ("iso_n", "#7570b3", "iso: n ∝ $p^2$")]:
    e = ESC[ESC.ley == ley]
    if len(e):
        ax.plot(e.p, e.E, "o-", color=c, ms=4, lw=1, label=nom)
ax.axhline(1, color="k", ls="--", lw=1, label="E = 1")
ax.set_ylim(0, 1.2)
ax.set_xticks([1, 2, 4, 6, 8])
ax.set_xlabel("p (#procesos)")
ax.set_ylabel("Eficiencia $E = T_s/(p\\,T_p)$")
ax.legend(fontsize=7, loc="lower left")
fig.tight_layout()
guardar(fig, "F7_escalabilidad")

# F8: versiones beta
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
for v, c in [(1, "#e7298a"), (2, "#7570b3"), (3, "#d95f02")]:
    b = BET[BET.version == v]
    if len(b) == 0:
        continue
    axs[0].errorbar(b.p, b["T"], yerr=barra(b), fmt="o-", color=c, ms=4, lw=1, capsize=2, label=f"beta {v}")
    axs[1].plot(b.p, b["T"].iloc[0] / b["T"], "o-", color=c, ms=4, lw=1, label=f"beta {v}")
if np.isfinite(T_base):
    axs[0].axhline(T_base, color="k", ls=":", lw=1, label="código base (secuencial)")
axs[0].set_xscale("log", base=2)
axs[0].set_yscale("log")
eje_p(axs[0])
axs[0].set_ylabel("$T_{ej}$ (s)")
axs[0].legend(fontsize=7)
axs[1].plot([1, 8], [1, 8], "k--", lw=1, label="S = p")
axs[1].set_xticks(P)
axs[1].set_xlabel("p (#procesos)")
axs[1].set_ylabel("Speedup relativo $T(1)/T(p)$")
axs[1].legend(fontsize=7)
fig.tight_layout()
guardar(fig, "F8_betas")

# F9: costo por par del computo local vs log2(n_tr/p)
fig, ax = plt.subplots(figsize=(3.5, 2.8))
for fac in FAC_OK:
    r = R[R.fac == fac]
    ax.plot(np.log2(r.n_q), r.c1 * 1e9, "o", color=COLOR[fac], ms=4, label=NOMBRE[fac])
xx = np.linspace(np.log2(R.n_q.min()), np.log2(R.n_q.max()), 50)
ax.plot(xx, (cd + cs * xx) * 1e9, "k-", lw=1, label="ajuste $c_d + c_s\\log_2(n_{tr}/p)$")
ax.set_xlabel("$\\log_2(n_{tr}/p)$")
ax.set_ylabel("$T_{comp_1}/(n_{te}\\,n_{tr}/p)$ (ns por par)")
ax.legend(fontsize=7)
fig.tight_layout()
guardar(fig, "F9_costo_por_par")

# F10: precision frente a n en los 4 tamanos de la serie fuerte (es la misma para todo p)
fig, ax = plt.subplots(figsize=(3.5, 2.8))
if len(PREC):
    ax.plot(PREC.n, PREC.acc, "o-", color="k", ms=4, lw=1)
    for _, f in PREC.iterrows():   # etiqueta arriba a la derecha (la del primer punto, abajo)
        ax.annotate(f"{NOMBRE[f.fac]}: {fmt(f.acc, 4)}", (f.n, f.acc), textcoords="offset points",
                    xytext=(5, -11) if f.n == PREC.n.min() else (3, 6), fontsize=7)
    ax.set_ylim(min(0.95, PREC.acc.min() - 0.005), 1.0)
    ax.set_xlim(PREC.n.min() / 1.6, PREC.n.max() * 4)        # margen para las etiquetas
else:
    sin_datos(ax, "")
ax.set_xscale("log")
ax.set_xlabel("n = $n_{tr} + n_{te}$ (imágenes)")
ax.set_ylabel("Precisión (accuracy)")
ax.set_title("Igual para todo p (betas 3 y 3b)", fontsize=8)
fig.tight_layout()
guardar(fig, "F10_precision")

# F11: beta 3 (gather) frente a 3b (reduce): T_ej (modo A) y pasos 5 a 7 (modo B)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, fac in zip(axs, [1, 16]):
    hay = False
    for RR, nom, c in [(R, "3 (gather)", "#d95f02"), (R3b, "3b (reduce)", "#1b9e77")]:
        r = RR[RR.fac == fac].sort_values("p")
        if len(r) == 0:
            continue
        hay = True
        ax.errorbar(r.p, r["T"], yerr=barra(r), fmt="o-", color=c, ms=4, lw=1, capsize=2,
                    label=f"$T_{{ej}}$, beta {nom}")
        ax.errorbar(r.p, r.T_57, yerr=barra(r, "T_57"), fmt="s--", color=c, mfc="white", ms=4, lw=1,
                    capsize=2, label=f"pasos 5 a 7, beta {nom}")
    if not hay:
        sin_datos(ax, NOMBRE[fac])
        continue
    n_tr, n_te = TAM[fac]
    ax.set_yscale("log")
    eje_p(ax)
    ax.set_ylabel("Tiempo (s)")
    ax.set_title(f"{NOMBRE[fac]} (n = {n_tr + n_te})", fontsize=9)
h, l = leyenda(axs)
fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0.13, 1, 1))
guardar(fig, "F11_gather_vs_reduccion")


# ===========================================================================
# TABLAS (LaTeX, booktabs)
# ===========================================================================
def tabla_resultados(fac):
    # T_ej: mediana +- la mitad del rango intercuartil (o del rango min-max con menos de 5 repeticiones)
    r = R[R.fac == fac].sort_values("p")
    lin = []
    for _, f in r.iterrows():
        dt = "--" if np.isnan(f.dTdp) else fmt(f.dTdp * 1e3, 2)
        lin.append(f"{int(f.p)} & {fmt(f['T'] * 1e3, 2)} $\\pm$ {fmt(f.T_err * 1e3, 2)} & {fmt(f.T_comp * 1e3, 2)} & "
                   f"{fmt(f.T_comm * 1e3, 2)} & {fmt(f.G, 1)} & {fmt(f.S, 2)} & {fmt(f.E, 2)} & {dt} & "
                   f"{fmt(f.GFLOPs, 2)} & {fmt(f.desb, 2)} \\\\")
    return "\n".join(lin) if lin else "% sin datos"


for fac in FACS:
    with open(f"{TAB}/T4_{NOMBRE[fac].replace('/', '')}.tex", "w") as f:
        f.write(tabla_resultados(fac))

with open(f"{TAB}/T3_tamanos.tex", "w") as f:
    for fac in FAC_OK:
        r = R[R.fac == fac].iloc[0]
        man, ex = f"{r.FLOP:.2e}".split("e")
        f.write(f"{TNOM[fac]} & {miles(r.n_tr)} & {miles(r.n_te)} & {miles(r.n_tr * r.n_te)} & "
                f"${man.replace('.', '{,}')}\\cdot 10^{{{int(ex)}}}$ & {fmt(r.acc, 4)} & {fmt(r.Ts * 1e3, 2)} \\\\\n")

with open(f"{TAB}/T6_popt.tex", "w") as f:
    for _, po in POPT.iterrows():
        f.write(f"{TNOM[po.fac]} & {miles(po.n_tr)} & {miles(po.n_te)} & {fmt(po.p_examen, 1)} & {fmt(po.p_num, 1)} & "
                f"{int(po.p_exp)} \\\\\n")

with open(f"{TAB}/T2_betas.tex", "w") as f:
    for v in [1, 2, 3]:
        b = BET[BET.version == v]
        if len(b) == 0:
            continue
        t1 = b[b.p == 1]["T"].iloc[0] if (b.p == 1).any() else np.nan
        t8 = b[b.p == 8]["T"].iloc[0] if (b.p == 8).any() else np.nan
        f.write(f"beta {v} & {fmt(b.acc.mean(), 4)} & {fmt(t1 * 1e3, 1)} & {fmt(t8 * 1e3, 1)} & {fmt(t1 / t8, 2)} \\\\\n")

with open(f"{TAB}/T_escalabilidad.tex", "w") as f:
    for _, e in ESC.iterrows():
        nom = "débil ($n \\propto \\sqrt{p}$)" if e.ley == "debil_n" else "iso ($n \\propto p^2$)"
        f.write(f"{nom} & {int(e.p)} & {miles(e.n_tr + e.n_te)} & {miles(e.n_tr)} & {fmt(e['T'] * 1e3, 1)} & "
                f"{fmt(e.Ts * 1e3, 1)} & {fmt(e.E, 2)} & {fmt(e.Ew, 2)} \\\\\n")

with open(f"{TAB}/T_pingpong.tex", "w") as f:
    # variante & alfa (us, m <= 100) & beta (ns por dato float64, m >= 1e4) & ancho de banda 8 bytes/beta
    # (GB/s) & error relativo maximo del modelo alfa + m beta en todos los m medidos (%)
    for var, nom in [("bufer", "búfer (\\texttt{Send}/\\texttt{Recv})"),
                     ("objetos", "objetos (\\texttt{send}/\\texttt{recv}, pickle)")]:
        if var in PP:
            c = PP[var]
            f.write(f"{nom} & {fmt(c['alfa'] * 1e6, 2)} & {fmt(c['beta'] * 1e9, 3)} & {fmt(c['GBs'], 1)} & "
                    f"{fmt(100 * c['err_rel_max'], 0)} \\\\\n")

# ---------------------------------------------------------------------------
# Resumen numerico
# ---------------------------------------------------------------------------
res = dict(estadistico="mediana de las repeticiones; barras: rango intercuartil con 5 o mas repeticiones, "
                       "minimo y maximo con menos",
           constantes=CONST, R2=R2, R2_log=R2_log, err_rel_medio=float(R.err_rel.mean()),
           err_rel_max=float(R.err_rel.max()), T_base=T_base,
           c1_por_fac={str(f): float(np.nanmedian(R[R.fac == f].c1)) for f in FAC_OK},
           c1_por_p={str(p): float(np.nanmedian(R[(R.p == p) & (R.fac >= 1)].c1)) for p in P
                     if ((R.p == p) & (R.fac >= 1)).any()},
           popt=POPT.to_dict(orient="records"),
           fuerte=R.drop(columns=["pares_q"]).to_dict(orient="records"),
           reduccion=R3b.to_dict(orient="records"),
           escalabilidad=ESC.to_dict(orient="records"),
           control=CTL.to_dict(orient="records"),
           betas=BET.to_dict(orient="records"),
           precision=PREC.to_dict(orient="records"),
           pingpong=PP,
           repeticiones=dict(min=int(MED.reps.min()), max=int(MED.reps.max())),
           carga="nucleos que usaban otros programas justo antes de cada mpirun (la mide el driver)",
           carga_media=float(df.carga.mean()), carga_max=float(df.carga.max()),
           cpu=dict(descripcion="minimo entre procesos de tiempo de CPU / tiempo de pared en la corrida medida",
                    minimo=float(df.cpu.min()), mediana=float(df.cpu.median()),
                    filas_bajo_0_9=int((df.cpu < 0.9).sum()), filas=int(df.cpu.notna().sum())),
           avisos=AVISOS)
with open(f"{DIR}/resumen.json", "w") as f:
    json.dump(res, f, indent=1, default=float)

print("Constantes:", {kk: f"{vv:.3e}" for kk, vv in CONST.items()})
print(f"R2 = {R2:.4f}  R2(log) = {R2_log:.4f}  error relativo medio = {R.err_rel.mean():.3f}")
print(POPT[["fac", "p_examen", "p_teo", "p_num", "p_exp"]])
print(R[["fac", "p", "T", "T_comp", "T_comm", "S", "E", "GFLOPs", "T_teo"]].round(4).to_string())
print(R3b[["fac", "p", "T", "T_57", "S", "E"]].round(4).to_string())
print(ESC.round(4).to_string())
print(CTL.round(4).to_string())
print(BET.round(4).to_string())
for var, c in PP.items():
    print(f"ping-pong ({var}): alfa = {c['alfa'] * 1e6:.2f} us, beta = {c['beta'] * 1e9:.3f} ns por dato "
          f"({c['GBs']:.1f} GB/s); error relativo maximo {100 * c['err_rel_max']:.0f} % (m = {c['m_err_max']}), "
          f"{100 * c['err_rel_max_m_grandes']:.0f} % con m >= {M_BETA:.0e}")
if AVISOS:
    print(f"{len(AVISOS)} avisos (tambien en resumen.json)")
