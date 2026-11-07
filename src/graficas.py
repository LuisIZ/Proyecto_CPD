"""Genera las gráficas y tablas del informe a partir de las mediciones.

Toma la mediana de las repeticiones. T_s es secuencial_vectorizado.py, para no
mezclar el efecto de la vectorización con el del paralelismo.

Uso: python graficas.py [--entrada ../results/final]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import nnls

from lanzador import RAIZ

# Paleta categórica en orden fijo: cada n conserva su color en todas las figuras.
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARCADORES = ["o", "s", "^", "D", "v", "P"]
TINTA, TINTA_2, REJILLA, FONDO = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
VERSIONES = ["gather", "reduccion"]
NOMBRES = {"gather": "Con gather (Beta 2)", "reduccion": "Con reducción (Beta 3)"}
METRICAS = ["t_total", "t_comp", "t_comm", "t_comm_max", "t_recoleccion", "t_seleccion",
            "t_comp_paralelo", "flop", "accuracy"]
N_ORIGINAL = 1797

# Pares (n, p) de la sección 6 del plan, con n/p² y n/(p log2 p) casi constantes.
SERIES_ISO = {
    r"$n \propto p^2$": [(450, 4), (1797, 8), (7200, 16)],
    r"$n \propto p \log_2 p$": [(450, 4), (1350, 8), (3600, 16)],
}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TINTA_2, "axes.labelcolor": TINTA_2,
    "axes.titlesize": 10, "axes.titlecolor": TINTA, "xtick.color": TINTA_2,
    "ytick.color": TINTA_2, "axes.grid": True, "grid.color": REJILLA,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.facecolor": FONDO,
})


def leer(ruta):
    """Devuelve {(version, n, p): {metrica: mediana}}."""
    grupos = defaultdict(lambda: defaultdict(list))
    with open(ruta, encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            clave = (fila["version"], int(fila["n"]), int(fila["p"]))
            for metrica in METRICAS:
                grupos[clave][metrica].append(float(fila[metrica]))
    return {c: {m: median(v) for m, v in g.items()} for c, g in grupos.items()}


def leer_secuencial(ruta):
    """Devuelve {n: T_s} con la mediana de las repeticiones."""
    tiempos = defaultdict(list)
    with open(ruta, encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            tiempos[int(fila["n"])].append(float(fila["t_total"]))
    return {n: median(t) for n, t in tiempos.items()}


def valores(datos, version, n, metrica):
    """Puntos (p, valor) de una serie, ordenados por p."""
    puntos = sorted((p, d[metrica]) for (v, m, p), d in datos.items()
                    if v == version and m == n)
    return np.array([x[0] for x in puntos]), np.array([x[1] for x in puntos])


def speedup(datos, t_s, version, n):
    p, t = valores(datos, version, n, "t_total")
    return p, t_s[n] / t


def ejes_p(ax, valores_p):
    ax.set_xscale("log", base=2)
    ax.set_xticks(valores_p, [str(p) for p in valores_p])
    ax.minorticks_off()
    ax.set_xlabel("Procesos ($p$)")


def serie(ax, x, y, i, etiqueta):
    ax.plot(x, y, color=COLORES[i], marker=MARCADORES[i], label=etiqueta)


def leyenda_comun(fig, ejes, columnas):
    manejadores, etiquetas = ejes.get_legend_handles_labels()
    fig.legend(manejadores, etiquetas, loc="lower center", ncol=columnas,
               bbox_to_anchor=(0.5, -0.07), labelcolor=TINTA_2)


def guardar(fig, salida, nombre):
    fig.savefig(salida / f"{nombre}.pdf")
    fig.savefig(salida / f"{nombre}.png", dpi=170)
    plt.close(fig)


def fig_tiempos(datos, ns, ps, salida):
    """Tiempo total, de cómputo y de comunicación, por versión."""
    fig, ejes = plt.subplots(2, 3, figsize=(9, 5.2), sharex=True, sharey=True)
    titulos = {"t_total": "Ejecución", "t_comp": "Cómputo", "t_comm": "Comunicación"}
    for fila, version in enumerate(VERSIONES):
        for columna, (metrica, titulo) in enumerate(titulos.items()):
            ax = ejes[fila, columna]
            for i, n in enumerate(ns):
                serie(ax, *valores(datos, version, n, metrica), i, f"$n={n}$")
            ejes_p(ax, ps)
            ax.set_yscale("log")
            ax.set_title(f"{titulo}: {NOMBRES[version].lower()}")
            if columna == 0:
                ax.set_ylabel("Tiempo (s)")
    fig.tight_layout()
    leyenda_comun(fig, ejes[0, 0], len(ns))
    guardar(fig, salida, "tiempos")


def fig_precision(datos, salida):
    """Precisión frente a n, solo con datos originales. No depende de p."""
    ns = sorted({c[1] for c in datos if c[0] == "gather" and c[1] <= N_ORIGINAL})
    precision = [datos[("gather", n, min(c[2] for c in datos))]["accuracy"] for n in ns]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(ns, precision, color=COLORES[0], marker=MARCADORES[0])
    for n, a in zip(ns, precision):
        ax.annotate(f"{a:.4f}", (n, a), xytext=(0, 7), textcoords="offset points",
                    ha="center", color=TINTA_2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlabel("Tamaño de los datos ($n$)")
    ax.set_ylabel("Precisión")
    ax.margins(y=0.25)
    fig.tight_layout()
    guardar(fig, salida, "precision")


def terminos(version, n, p):
    """Columnas del modelo normalizado de la Sección 2.5 del informe."""
    log_p = np.log2(p)
    if version == "gather":
        return np.column_stack([n**2 / p, n * p, n * log_p, log_p])
    return np.column_stack([n**2 / p, n * log_p, log_p])


def terminos_comp(version, n, p):
    if version == "gather":
        return np.column_stack([n**2 / p, n * p])
    return np.column_stack([n**2 / p, n * np.log2(p)])


def terminos_comm(version, n, p):
    """El término n (p - 1) es el del gather en el modelo de árbol binomial."""
    log_p = np.log2(p)
    if version == "gather":
        return np.column_stack([n * log_p, log_p, n * (p - 1)])
    return np.column_stack([n * log_p, log_p])


def ajustar(datos, version, metrica="t_total", funcion=terminos, p_minimo=1):
    """Ajusta c_i >= 0 por mínimos cuadrados sobre el error relativo."""
    claves = sorted(c for c in datos if c[0] == version and c[2] >= p_minimo)
    n = np.array([c[1] for c in claves], dtype=float)
    p = np.array([c[2] for c in claves], dtype=float)
    t = np.array([datos[c][metrica] for c in claves])
    A = funcion(version, n, p)
    coeficientes, _ = nnls(A / t[:, None], np.ones_like(t))
    return coeficientes, float(np.mean(np.abs(A @ coeficientes - t) / t))


def predecir(version, coeficientes, n, p, funcion=terminos):
    return (funcion(version, float(n), np.atleast_1d(np.asarray(p, dtype=float)))
            @ coeficientes)


def p_optimo(version, coeficientes, n):
    """Fórmulas de p* de la Sección 2.5. Devuelve inf si el término es nulo."""
    if version == "gather":
        c1, c2 = coeficientes[0], coeficientes[1]
        return np.sqrt(c1 * n / c2) if c2 > 0 else np.inf
    c1, c3, c4 = coeficientes
    denominador = c3 * n + c4
    return np.log(2) * c1 * n**2 / denominador if denominador > 0 else np.inf


def cruce(version, ajuste_comp, ajuste_comm, n):
    """Primer p en que la comunicación del modelo alcanza al cómputo."""
    malla = np.geomspace(1, 4096, 800)
    comp = predecir(version, ajuste_comp[0], n, malla, terminos_comp)
    comm = predecir(version, ajuste_comm[0], n, malla, terminos_comm)
    cruzan = np.nonzero(comm >= comp)[0]
    return malla[cruzan[0]] if len(cruzan) else np.inf


def fig_teoria(datos, ajustes, ns, ps, salida):
    """Mediciones (puntos) y modelo normalizado (líneas)."""
    fig, ejes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    malla = np.geomspace(min(ps), max(ps), 60)
    for ax, version in zip(ejes, VERSIONES):
        for i, n in enumerate(ns):
            p, t = valores(datos, version, n, "t_total")
            ax.plot(malla, predecir(version, ajustes[version][0], n, malla), color=COLORES[i])
            ax.plot(p, t, linestyle="none", color=COLORES[i], marker=MARCADORES[i],
                    label=f"$n={n}$")
        ejes_p(ax, ps)
        ax.set_yscale("log")
        ax.set_title(f"{NOMBRES[version]}: error medio {100 * ajustes[version][1]:.0f} %")
    ejes[0].set_ylabel("Tiempo de ejecución (s)")
    fig.tight_layout()
    leyenda_comun(fig, ejes[0], len(ns))
    guardar(fig, salida, "teoria")


def fig_teoria_fases(datos, fases, ns, ps, salida):
    """Cómputo y comunicación medidos, con su ajuste por separado."""
    fig, ejes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True)
    malla = np.geomspace(min(ps), max(ps), 60)
    for fila, version in enumerate(VERSIONES):
        for columna, (metrica, funcion, titulo) in enumerate(
                [("t_comp", terminos_comp, "Cómputo"), ("t_comm", terminos_comm, "Comunicación")]):
            ax = ejes[fila, columna]
            coeficientes, error = fases[version][metrica]
            for i, n in enumerate(ns):
                p, t = valores(datos, version, n, metrica)
                inicio = 0 if metrica == "t_comp" else 1
                ax.plot(p[inicio:], t[inicio:], linestyle="none", color=COLORES[i],
                        marker=MARCADORES[i], label=f"$n={n}$")
                curva = malla if metrica == "t_comp" else malla[malla >= 2]
                ax.plot(curva, predecir(version, coeficientes, n, curva, funcion),
                        color=COLORES[i])
            ejes_p(ax, ps)
            ax.set_yscale("log")
            ax.set_title(f"{titulo}, {NOMBRES[version].lower()}: error {100 * error:.0f} %")
            if columna == 0:
                ax.set_ylabel("Tiempo (s)")
    fig.tight_layout()
    leyenda_comun(fig, ejes[0, 0], len(ns))
    guardar(fig, salida, "teoria_fases")


def fig_speedup_eficiencia(datos, t_s, ns, ps, salida):
    fig, ejes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True, sharey="row")
    for columna, version in enumerate(VERSIONES):
        for i, n in enumerate(ns):
            p, s = speedup(datos, t_s, version, n)
            serie(ejes[0, columna], p, s, i, f"$n={n}$")
            serie(ejes[1, columna], p, s / p, i, f"$n={n}$")
        ejes[0, columna].plot(ps, ps, color=TINTA_2, linestyle="--", linewidth=1,
                              label="Ideal")
        ejes[0, columna].set_yscale("log", base=2)
        ejes[0, columna].yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ejes[0, columna].set_title(f"Speedup: {NOMBRES[version].lower()}")
        ejes[1, columna].set_title(f"Eficiencia: {NOMBRES[version].lower()}")
        ejes[1, columna].set_ylim(0, None)
        ejes_p(ejes[0, columna], ps)
        ejes_p(ejes[1, columna], ps)
    ejes[0, 0].set_ylabel("$S = T_s / T_p$")
    ejes[1, 0].set_ylabel("$E = S / p$")
    fig.tight_layout()
    leyenda_comun(fig, ejes[0, 0], len(ns) + 1)
    guardar(fig, salida, "speedup_eficiencia")


def fig_flops(datos, ns, ps, salida):
    """FLOP por segundo: con el tiempo total y solo con el de cómputo."""
    fig, ejes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True, sharey=True)
    filas = [("t_total", "$F / T_p$"), ("t_comp_paralelo", r"$F / \max\, t_{comp}$")]
    for fila, (metrica, titulo) in enumerate(filas):
        for columna, version in enumerate(VERSIONES):
            ax = ejes[fila, columna]
            for i, n in enumerate(ns):
                p, t = valores(datos, version, n, metrica)
                _, f = valores(datos, version, n, "flop")
                serie(ax, p, f / t / 1e9, i, f"$n={n}$")
            ejes_p(ax, ps)
            ax.set_ylim(0, None)
            ax.set_title(f"{titulo}: {NOMBRES[version].lower()}")
            if columna == 0:
                ax.set_ylabel("GFLOP/s")
    fig.tight_layout()
    leyenda_comun(fig, ejes[0, 0], len(ns))
    guardar(fig, salida, "flops")


def fig_granularidad(datos, ns, ps, salida):
    """G = T_comp / T_comm. Con G < 1 domina la comunicación."""
    fig, ejes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, version in zip(ejes, VERSIONES):
        for i, n in enumerate(ns):
            p, comp = valores(datos, version, n, "t_comp")
            _, comm = valores(datos, version, n, "t_comm")
            serie(ax, p[1:], comp[1:] / comm[1:], i, f"$n={n}$")
        ax.axhline(1, color=TINTA_2, linestyle="--", linewidth=1, label="$G = 1$")
        ejes_p(ax, ps[1:])
        ax.set_yscale("log")
        ax.set_title(NOMBRES[version])
    ejes[0].set_ylabel(r"$G = T_{comp} / T_{comm}$")
    fig.tight_layout()
    leyenda_comun(fig, ejes[0], len(ns) + 1)
    guardar(fig, salida, "granularidad")


def eficiencias(datos, t_s, ajustes):
    """{version: {condicion: [(n, p, E medida, E del modelo)]}}."""
    resultado = {}
    for version in VERSIONES:
        resultado[version] = {}
        for condicion, pares in SERIES_ISO.items():
            puntos = []
            for n, p in pares:
                if (version, n, p) in datos and n in t_s:
                    e = t_s[n] / (p * datos[(version, n, p)]["t_total"])
                    c = ajustes[version][0]
                    modelo = predecir(version, c, n, 1)[0] / (p * predecir(version, c, n, p)[0])
                    puntos.append((n, p, e, modelo))
            resultado[version][condicion] = puntos
    return resultado


def fig_isoeficiencia(series, salida):
    """E medida (continua) y del modelo (punteada) en cada condición."""
    fig, ejes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, version in zip(ejes, VERSIONES):
        for i, (condicion, puntos) in enumerate(series[version].items()):
            if puntos:
                p = [x[1] for x in puntos]
                serie(ax, p, [x[2] for x in puntos], i, condicion)
                ax.plot(p, [x[3] for x in puntos], color=COLORES[i], linestyle=":")
        ejes_p(ax, [4, 8, 16])
        ax.set_ylim(0, 1.05)
        ax.set_title(NOMBRES[version])
    ejes[0].set_ylabel("Eficiencia $E$")
    manejadores, etiquetas = ejes[0].get_legend_handles_labels()
    manejadores.append(Line2D([], [], color=TINTA_2, linestyle=":"))
    etiquetas.append("Modelo normalizado")
    fig.tight_layout()
    fig.legend(manejadores, etiquetas, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.08), labelcolor=TINTA_2)
    guardar(fig, salida, "isoeficiencia")


def fig_paso6(datos, ps, salida, n=N_ORIGINAL):
    """Costo de los pasos 5 a 7 en las dos versiones."""
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, version in enumerate(VERSIONES):
        p, recoleccion = valores(datos, version, n, "t_recoleccion")
        _, seleccion = valores(datos, version, n, "t_seleccion")
        serie(ax, p, recoleccion + seleccion, i, NOMBRES[version])
    ejes_p(ax, ps)
    ax.set_yscale("log")
    ax.set_ylabel("Tiempo de los pasos 5 a 7 (s)")
    ax.legend(labelcolor=TINTA_2)
    fig.tight_layout()
    guardar(fig, salida, "paso6")


def ajustar_pingpong(ruta):
    """Ajusta t(m) = a + b m, pesando por 1/t para no ignorar mensajes chicos."""
    with open(ruta, encoding="utf-8") as archivo:
        filas = [(float(f["tamano"]), float(f["t_mensaje"])) for f in csv.DictReader(archivo)]
    m, t = np.array(filas).T
    A = np.column_stack([np.ones_like(m), m])
    return nnls(A / t[:, None], np.ones_like(t))[0]


def escribir_resumen(datos, t_s, ajustes, fases, series, ns, carpeta, salida):
    lineas = ["# Resumen de mediciones", ""]
    if (carpeta / "pingpong.csv").exists():
        a, b = ajustar_pingpong(carpeta / "pingpong.csv")
        lineas += [f"Ping-pong: a = {a:.3e} s, b = {b:.3e} s por dato float64.", ""]
    if (carpeta / "base.csv").exists():
        with open(carpeta / "base.csv", encoding="utf-8") as archivo:
            base = [float(f["t_total"]) for f in csv.DictReader(archivo)]
        lineas += [f"Código base, n = {N_ORIGINAL}: mediana {median(base):.4f} s "
                   f"({len(base)} repeticiones).", ""]
    lineas += ["## T_s frente a p = 1", "", "| n | T_s (s) | gather p = 1 | reducción p = 1 |",
               "|---|---|---|---|"]
    for n in ns:
        lineas.append(f"| {n} | {t_s[n]:.4f} | {datos[('gather', n, 1)]['t_total']:.4f} | "
                      f"{datos[('reduccion', n, 1)]['t_total']:.4f} |")
    lineas.append("")
    for version in VERSIONES:
        coeficientes, error = ajustes[version]
        nombres = ["c1", "c2", "c3", "c4"] if version == "gather" else ["c1", "c3", "c4"]
        comp, comm = fases[version]["t_comp"], fases[version]["t_comm"]
        lineas.append(f"## {NOMBRES[version]}")
        lineas.append("Coeficientes (s): " + ", ".join(
            f"{x} = {y:.3e}" for x, y in zip(nombres, coeficientes)))
        lineas.append(f"Error relativo medio del ajuste: {100 * error:.1f} %")
        lineas.append(f"Ajuste de T_comp: {np.array2string(comp[0], precision=3)}, "
                      f"error {100 * comp[1]:.1f} %")
        lineas.append(f"Ajuste de T_comm (p > 1): {np.array2string(comm[0], precision=3)}, "
                      f"error {100 * comm[1]:.1f} %")
        lineas += ["", "| n | p* estimado | cruce comp/comm (inf: no se cruzan) | mejor p medido | speedup máximo |",
                   "|---|---|---|---|---|"]
        for n in ns:
            p, t = valores(datos, version, n, "t_total")
            _, s = speedup(datos, t_s, version, n)
            lineas.append(f"| {n} | {p_optimo(version, coeficientes, n):.1f} | "
                          f"{cruce(version, comp, comm, n):.1f} | {p[np.argmin(t)]} | "
                          f"{s.max():.2f} |")
        lineas += ["", "| Condición | (n, p) | E medida | E del modelo |", "|---|---|---|---|"]
        for condicion, puntos in series[version].items():
            for n, p, e, modelo in puntos:
                lineas.append(f"| {condicion} | ({n}, {p}) | {e:.3f} | {modelo:.3f} |")
        lineas.append("")
    lineas += ["## Precisión", "", "| n | accuracy |", "|---|---|"]
    for n in ns:
        if n <= N_ORIGINAL:
            lineas.append(f"| {n} | {datos[('gather', n, 1)]['accuracy']:.4f} |")
    (salida / "resumen.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrada", default=str(RAIZ / "results" / "final"))
    args = parser.parse_args()

    carpeta = Path(args.entrada)
    datos = leer(carpeta / "mediciones.csv")
    t_s = leer_secuencial(carpeta / "secuencial.csv")
    salida = carpeta / "figuras"
    salida.mkdir(exist_ok=True)

    ns = sorted({c[1] for c in datos})
    ps = sorted({c[2] for c in datos})
    ajustes = {v: ajustar(datos, v) for v in VERSIONES}
    fases = {v: {"t_comp": ajustar(datos, v, "t_comp", terminos_comp),
                 "t_comm": ajustar(datos, v, "t_comm", terminos_comm, p_minimo=2)}
             for v in VERSIONES}
    series = eficiencias(datos, t_s, ajustes)

    fig_tiempos(datos, ns, ps, salida)
    fig_precision(datos, salida)
    fig_teoria(datos, ajustes, ns, ps, salida)
    fig_teoria_fases(datos, fases, ns, ps, salida)
    fig_speedup_eficiencia(datos, t_s, ns, ps, salida)
    fig_flops(datos, ns, ps, salida)
    fig_granularidad(datos, ns, ps, salida)
    fig_isoeficiencia(series, salida)
    fig_paso6(datos, ps, salida)
    escribir_resumen(datos, t_s, ajustes, fases, series, ns, carpeta, salida)
    print(f"Figuras y resumen en {salida}")


if __name__ == "__main__":
    main()
