# Caso 5: KNN en paralelo con MPI (mpi4py) — Grupo 7

CS4052 Computación Paralela y Distribuida, UTEC 2026-II.
Integrantes: Barzola Estrella, Johar Jared; Huaman Casma, Leonardo Fabian; Izaguirre Zavaleta, Luis Fernando.

## Archivos

| Archivo | Qué es |
|---|---|
| `knn_digits_sec.py` | Código secuencial del profesor (sin cambios) |
| `knn_mpi_beta1.py` | Beta 1: traducción directa del algoritmo del DAG (bcast, scatter, gather); distancias con bucle de Python como el código base |
| `knn_mpi_beta2.py` | Beta 2: distancias con NumPy por punto de prueba; con ella empezaron la medición por fases y el CSV (hoy las tres betas se miden igual) |
| `knn_mpi_beta3.py` | Beta 3 (final): distancias por bloques con producto de matrices; tamaño variable del problema; versión secuencial para T_s; con `--reduccion`, la beta 3b (`comm.reduce`) |
| `validar.py` | Validación: predicciones idénticas al código base y a la versión secuencial (`validacion.txt`) |
| `ping_pong.py` | Latencia α y tiempo por dato β entre dos procesos (`pingpong.csv`) |
| `pruebas_extra.py` | Pruebas aparte citadas en el informe: empates, costo por par de cada paso y contención de hardware |
| `correr_experimentos.py` | Barrido completo: entorno, paso 0 (validación, ping-pong, pruebas aparte) y malla de mediciones |
| `correr_experimentos.sh` | Envoltorio de una línea de `correr_experimentos.py` (para los comandos de antes) |
| `analisis.py` | Tablas, figuras y ajuste del modelo teórico a partir de `tiempos.csv` y `pingpong.csv` |

## Requisitos

- Open MPI (por ejemplo `brew install open-mpi`) y Python 3.11
- `pip install -r requirements.txt`

## Cómo ejecutar

```bash
# una corrida con 4 procesos y validación contra el código secuencial del profesor
mpirun -np 4 python knn_mpi_beta1.py --validar
mpirun -np 4 python knn_mpi_beta2.py --validar
mpirun -np 4 python knn_mpi_beta3.py --validar
mpirun -np 4 python knn_mpi_beta3.py --validar --reduccion        # beta 3b

# beta 3 con un problema 16 veces más grande, y su versión secuencial (T_s)
mpirun -np 8 python knn_mpi_beta3.py --fac 16
python knn_mpi_beta3.py --secuencial --fac 16

# escalabilidad con un solo n: débil (n ~ raiz(p)) e isoeficiencia (n ~ p^2)
mpirun -np 4 python knn_mpi_beta3.py --ley debil_n
mpirun -np 4 python knn_mpi_beta3.py --ley iso_n
python knn_mpi_beta3.py --secuencial --ley iso_n --p_ley 4    # su T_s

# serie de control (n_te fijo, n_tr = 1437 p), citada en la sección de diferencias del informe
mpirun -np 4 python knn_mpi_beta3.py --ley control

# validación completa, ping-pong y pruebas aparte
python validar.py resultados/validacion.txt
mpirun -np 2 python ping_pong.py --csv resultados/pingpong.csv
for p in 1 4 8; do mpirun -np $p python pruebas_extra.py; done

# figuras y tablas a partir de los datos medidos (por defecto, resultados/: los del informe)
python analisis.py
```

Cada beta acepta `--rep r` (índice de la repetición), `--calentamientos W` (rondas de calentamiento),
`--carga x` (la pone el driver) y `--csv archivo` (agrega una fila por modo). Si se corren a mano, conviene fijar un hilo por proceso:
`export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
Las betas también corren en Windows (con `os.getloadavg` ausente, la carga queda en NaN).

## Barrido completo de experimentos

Con la computadora libre (sin OneDrive, navegador, etc.), enchufada y con el entorno del proyecto
activado:

```bash
caffeinate -i python correr_experimentos.py                  # escribe en resultados_nuevos/ (unos 35 min)
python analisis.py --resultados resultados_nuevos

caffeinate -i python correr_experimentos.py otra --reps 6    # 6 repeticiones (unos 45 min)
python correr_experimentos.py --plan                         # solo muestra la malla y cuánto tardaría
python correr_experimentos.py prueba_rapida --rapido         # malla chica para probar (unos 5 min)
PY=/ruta/a/python caffeinate -i ./correr_experimentos.sh     # lo mismo, con el envoltorio de antes
```

- **Hilos.** El driver fija `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1` y los pasa a `mpirun` con `-x`, para que el speedup sea el de p procesos y
  no el de los hilos de BLAS.
- **Carpeta.** El primer argumento es la carpeta de salida (por defecto `resultados_nuevos`). Si ya
  tiene una corrida completa, no la pisa. Si tiene una parcial (por ejemplo, se cortó la luz), la
  retoma: salta las (versión, n_tr, n_te, p, rep) que ya están en `tiempos.csv`; si una quedó a
  medias (por ejemplo, solo con el modo B), quita sus filas y la mide de nuevo, para que no queden
  filas repetidas. No retoma un `tiempos.csv` que no sea de este barrido (con otras columnas o sin
  `malla.json`, como el de septiembre): pide otra carpeta.
- **Paso 0.** Primero escribe `entorno.txt` (fecha, CPU, núcleos de rendimiento y de eficiencia,
  versiones de Python, numpy, scikit-learn, mpi4py, Open MPI y macOS) y corre la validación
  (`validacion.txt`), el ping-pong (`pingpong.csv`) y `pruebas_extra.py` con 1, 4 y 8 copias
  (`pruebas_extra.txt`).
- **Fallos.** Cada `mpirun` tiene un tiempo máximo de 5 veces lo estimado (al menos 60 s) y un
  reintento; los fallos quedan en `fallos.log` y el barrido sigue. Si se pasa del tiempo, el driver
  mata a `mpirun` y a todos sus procesos (Open MPI deja cada proceso en su propio grupo, así que no
  basta con matar a `mpirun`). En pantalla se ve el avance con el tiempo que falta.
- **Máquina libre.** Antes de cada `mpirun`, el driver mide la carga externa (los núcleos que usan
  los programas ajenos al barrido, con `ps`) y la pasa a la beta, que la guarda en la columna `carga`.
  Al inicio avisa si pasa de 1,5 núcleos. Después de un `mpirun` de más de 20 s hace una pausa de 3 s.
- **Energía.** El driver lanza `caffeinate -i -w` (la Mac no se duerme mientras corre) y anota en
  `entorno.txt`, al inicio y al final, si estaba enchufada, el modo de bajo consumo y el estado
  térmico (`pmset`). Avisa si está a batería o en bajo consumo.
- El driver está hecho para macOS o Linux con Open MPI (usa `mpirun -x`, `ps` y `sysctl`).

### Malla

Cada configuración (versión, n_tr, n_te, p) se mide **una vez por repetición**, aunque esté en
varias series: `analisis.py` arma las series buscando por tamaño. Por ejemplo, n0 con p = 1 está
en la serie fuerte, la débil, la iso y la de control, y todas comparten esa misma línea base.

| Serie | Tamaños | p | Versiones |
|---|---|---|---|
| Fuerte (n fijo) | n/4 (359×90), n0 (1437×360), 4n (5748×1440), 16n (22 992×5760) | 1, 2, 4, 6, 8 | 3 y 3b |
| Betas 1 y 2 | n0 | 1, 2, 4, 6, 8 | 1 y 2 |
| Débil, n ∝ √p | n0, 2541, 3594 y 5082 imágenes (cada una con su p) | 1, 2, 4, 8 | 3 |
| Isoeficiencia, n ∝ k p² | n0, 4n, 16n y 115 008 imágenes (cada una con su p) | 1, 2, 4, 8 | 3 |
| Control (n_te fijo) | n_te = 360, n_tr = 1437 p | 1, 2, 4, 6, 8 | 3 |
| Secuencial (T_s, modo S) | n/4, n0, 4n, 16n, 115 008 y los de la débil | 1 | 3 |
| Código base del profesor | n0 | 1 | 0 |

Son 67 configuraciones con 4 repeticiones, salvo las de más de 10⁹ pares (115 008 imágenes, en
paralelo y en secuencial), que llevan 2: 264 `mpirun`. El T_s de cada tamaño se mide junto a
sus corridas en paralelo: antes en las repeticiones pares y después en las impares.

## Protocolo de medición

- Se mide con `MPI.Wtime()` en el maestro, desde antes del broadcast hasta después de calcular la
  precisión. La carga y la división de los datos quedan fuera, como en el código base.
- **Un `mpirun` por repetición.** Cada `mpirun` mide una sola repetición de una configuración:
  primero W rondas de calentamiento que se descartan y después una corrida medida por modo. Escribe
  una fila del CSV por modo.
- **Calentamiento.** Las primeras corridas de cada proceso son más lentas (con n/4 y p = 8, la
  cuarta todavía tarda el doble; con n0 y p = 8, un 35 % más). Por eso, en cada ronda de
  calentamiento se corre una vez cada modo, intercalados, y los dos llegan igual de calientes a la
  medición. W se fija por tiempo: el maestro cronometra la primera ronda (d por corrida) y
  W = ⌈0,25 s / d⌉, entre 1 y 30; W = 0 si n_tr·n_te > 2·10⁸, porque una corrida ya tarda varios
  segundos (`--calentamientos` lo fija a mano).
- **Modos.** En el modo A no hay barreras dentro de la región medida; de ahí sale T_ej (y S, E y
  FLOP/s). En el modo B hay un `comm.Barrier()` antes de cada marca, solo para separar las fases;
  de ahí salen T_bcast, T_scatter, T_comp, T_gather y T_comp2. El orden alterna con la repetición:
  A, B si la repetición es par y B, A si es impar (también en el calentamiento). El modo S es la
  versión secuencial (T_s), con la misma regla de calentamiento.
- **Orden.** Las repeticiones van afuera: primero la repetición 0 de toda la malla, después la 1,
  etc. En las repeticiones pares la malla se recorre hacia adelante y en las impares hacia atrás.
  Con un número **par** de repeticiones, la mediana de cada configuración cae en el mismo punto
  medio de la sesión para todas, así que una deriva lineal de la máquina (temperatura, otra carga)
  se cancela, y cada modo va primero en la mitad de las repeticiones. Con un número impar, la
  mediana es la repetición del medio y la deriva no se cancela: por eso son 4 (y 2), no 5 (y 3).
- **Estadístico.** `analisis.py` toma la **mediana** de las repeticiones de cada (versión, n_tr,
  n_te, p, modo). Las barras de error son el rango intercuartil; con menos de 5 repeticiones (las 4
  por defecto), el mínimo y el máximo. En las tablas T4, T_ej se escribe como mediana ± la mitad de
  ese rango. T_comm y T_comp son la suma de las medianas de sus fases, así que T_comm = T_bcast +
  T_scatter + T_gather también en las tablas; los pasos 5 a 7 (T_gather + T_comp2) son un solo
  intervalo de cada corrida y se toma la mediana de esa suma.

## Validación

`validar.py` corre cada beta con `--validar --calentamientos 0` y compara las predicciones de los
modos A y B. El resultado, con la fecha y la versión de cada paquete, queda en `validacion.txt`:

- n0 (datos originales): betas 1, 2, 3 y 3b con p = 1 a 8, contra el código base (`knn_predict` del
  profesor, copiada en las betas); la precisión tiene que ser 0,9833.
- n/4 y 4n: betas 3 y 3b con p = 4 y 8, contra la versión secuencial y contra el código base.
- 16n: betas 3 y 3b con p = 8, contra la versión secuencial. Con 1,3·10⁸ pares, el bucle de Python
  del código base tardaría unos 4 minutos.
- Serie débil (2541, 3594 y 5082 imágenes, cada una con su p): beta 3 contra la versión secuencial
  y el código base.
- Isoeficiencia: n0, 4n y 16n ya están arriba. El tamaño de 115 008 imágenes se omite, porque su
  versión secuencial sola tarda unos 3 minutos.
- Relleno: betas 3 y 3b con n_tr = 7 y n_te = 36, con p = 4 (bloques de 2, 2, 2 y 1 puntos, menos que
  k = 3) y p = 8 (un bloque vacío), contra la versión secuencial y el código base. Así se prueban los
  candidatos con distancia +∞.

Todas las versiones ordenan de forma estable, desempatan por (distancia, índice global) y votan
como `Counter.most_common`, así que el resultado no depende de p. Si un bloque tiene menos de k
puntos, los candidatos que faltan quedan con distancia +∞ y no ganan nunca (betas 2 y 3; en la
beta 1 la lista simplemente queda más corta).

## Ping-pong

`ping_pong.py` manda mensajes de m datos float64 entre dos procesos, con m de 1 a 10⁶ en escala
logarítmica, en dos variantes:

- con búfer: `comm.Send`/`comm.Recv` con arreglos de NumPy, sin serializar;
- con objetos de Python: `comm.send`/`comm.recv` con pickle, que es lo que usan las colectivas en
  minúscula de las betas.

Cada tamaño se mide 7 veces. Cada vez se cronometran varias idas y vueltas seguidas y se toma la
mitad (el tiempo de ida). Se guarda la mediana en `pingpong.csv`.

`analisis.py` usa el modelo T(m) = α + m·β para cada variante. Ninguna recta describe bien los seis
órdenes de magnitud de m (con pickle, el costo por dato cambia con el tamaño del mensaje), así que
se separan los dos parámetros para poder interpretarlos, no porque ajusten mejor: α es la
latencia (la mediana del tiempo de ida con m ≤ 100) y β es la pendiente por mínimos cuadrados con
m ≥ 10⁴, que es el rango de las colectivas de las betas. α, β, el ancho de banda (8 bytes/β) y el
error relativo máximo del modelo quedan en `resumen.json` y en `tablas/T_pingpong.tex`. En
`resumen.json` también está, como referencia, la recta ajustada con todos los m.

## Beta 3b: reducción en vez de gather

`knn_mpi_beta3.py --reduccion` cambia el gather de los candidatos y la mezcla del maestro por
`comm.reduce(C_q, op=mezcla, root=0)`; en el CSV, `version = 3b`. `mezcla(a, b)` deja, por cada punto
de prueba, los k candidatos de menor (distancia, índice global) de la unión de a y b, con el mismo
desempate que la versión con gather. Por eso las predicciones son idénticas. La operación es
asociativa y conmutativa, así que el resultado no depende de cómo se agrupen los procesos. Después,
el maestro solo vota.

La beta 3 con gather sigue siendo la versión principal, porque el enunciado nombra `comm.gather`. La
3b es la mejora.

Para objetos de Python, mpi4py hace la reducción con un árbol binomial sobre los rangos
(`mpi4py.rc.fast_reduce`). El maestro hace ⌈log₂ p⌉ mezclas. Los procesos que ya enviaron su parte
siguen recibiendo y mezclando, aunque eso no se use, así que el total es mayor que p − 1: con p = 8
son 12 mezclas, no 7. Ese trabajo extra ocurre después de que cada proceso aporta lo suyo, así que
no retrasa al maestro.

**Fases en el modo B.** Las columnas son las mismas en las dos versiones, pero cambia lo que mide cada una:

| Columna | Beta 3 (gather) | Beta 3b (reduce) |
|---|---|---|
| `T_gather` | `comm.gather` de los p·k candidatos | `comm.reduce`, incluidas las mezclas del árbol |
| `T_comp2` | mezcla de los p·k candidatos, votación y precisión | solo votación y precisión |

Por eso la figura F11 compara T_ej (modo A) y los pasos 5 a 7 (T_gather + T_comp2, modo B), que
miden lo mismo en las dos versiones.

## Estructura de `tiempos.csv`

Hay una fila por (configuración, repetición, modo):

| Columna | Qué es |
|---|---|
| `version` | 0 = código base del profesor, 1, 2, 3 o 3b |
| `n_tr`, `n_te` | tamaño de entrenamiento y de prueba |
| `p` | número de procesos (1 en el modo S) |
| `rep` | índice de la repetición |
| `modo` | A (sin barreras, T_ej), B (con barreras, fases) o S (secuencial) |
| `W` | rondas de calentamiento descartadas antes de la corrida medida (en cada una, una corrida por modo) |
| `T_ej` | tiempo de la región medida (s) |
| `T_bcast`, `T_scatter`, `T_gather` | colectivas (s) |
| `T_comp_max`, `T_comp_prom` | cómputo local (pasos 3 y 4): máximo y promedio entre procesos (s) |
| `T_comp2` | trabajo del maestro después del gather o del reduce (s) |
| `acc` | precisión |
| `carga` | carga externa: núcleos que usaban los programas ajenos al barrido justo antes del `mpirun` (la mide el driver con `ps`). Si la beta se corre a mano, el promedio de carga de 1 min al empezar (NaN si el sistema no lo da) |
| `cpu` | mínimo entre procesos de tiempo de CPU / tiempo de pared en la corrida medida: ~1 si ningún proceso fue desalojado por otro programa (NaN en el código base) |
| `hora` | hora en que terminó la medición |

Las columnas `ley` y `fac` de la versión anterior ya no están: las series se reconocen por
(n_tr, n_te).

## Salidas de `analisis.py`

- `figuras/F1_tiempos.pdf` a `F9_costo_por_par.pdf`, como antes, más `F10_precision.pdf` (precisión de los cuatro
  tamaños de la serie fuerte frente a n; es la misma para todo p) y `F11_gather_vs_reduccion.pdf`
  (betas 3 y 3b en n0 y 16n). Todas van también en PNG, a 200 dpi y con fondo blanco, en
  `figuras/png/`.
- `tablas/*.tex`: las de antes (`T2_betas`, `T3_tamanos`, `T4_*`, `T6_popt`, `T_escalabilidad`) y
  `T_pingpong`.
- `resumen.json`: constantes del modelo, p óptimo, series (fuerte, 3b, débil, iso, control y
  betas), precisión, α y β del ping-pong, carga externa, columna `cpu` y avisos.

Si falta una serie, `analisis.py` lo avisa y genera igual las figuras y las tablas, con huecos.
También avisa si hay filas repetidas (se queda con la última), si el CSV es del protocolo viejo, si
en más del 5 % de las corridas algún proceso tuvo menos del 90 % de la CPU o si la carga externa
mediana pasa de 1,5 núcleos.

## Mapeo del algoritmo distribuido (el del DAG) → código

| Paso del algoritmo | Código |
|---|---|
| 1. Broadcast de la data de prueba | `comm.bcast((X_test, k), root=0)` |
| 2. Scatter de la data de entrenamiento | `comm.scatter(bloques, root=0)` con `np.array_split` |
| 3–4. Distancias y k vecinos locales | `k_vecinos_locales(...)` |
| 5. Gather de los candidatos | `comm.gather(C_q, root=0)` |
| 6–7. k globales, votación y precisión | `mezclar_y_votar(...)` en el maestro |
| 5–6 en la beta 3b | `comm.reduce(C_q, op=mezcla, root=0)`; luego `votar(...)` en el maestro |

## Notas

- Los datos del informe final están en `resultados/`: se midieron el 11 de noviembre de 2026 con
  `correr_experimentos.py`, en una sola sesión de 33 min (ver `resultados/entorno.txt`), y las figuras
  y tablas salen de `python analisis.py`. Las mediciones de septiembre usaban otro protocolo
  (5 repeticiones seguidas en un mismo `mpirun`, con media ± desviación) y ya no se usan.
- El modelo normalizado usa latencia `log2(p)` para scatter y gather, como las colectivas en árbol
  de MPI: O(α log p + βm) (U2.4 y correo del profesor del 6 de noviembre de 2026).
