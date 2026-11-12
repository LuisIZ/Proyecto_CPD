# KNN en paralelo con MPI (mpi4py)

Proyecto de laboratorio de CS4052 Computación Paralela y Distribuida (UTEC, 2026-II), caso 5, Grupo 7: Johar Barzola, Leonardo Huaman y Luis Izaguirre. Paraleliza con mpi4py la clasificación KNN de los dígitos de `load_digits`.

## Contenido

| Carpeta | Qué hay |
|---|---|
| [`informe_final/`](informe_final) | Informe final ([`informe_final.pdf`](informe_final/informe_final.pdf), 17 páginas), su fuente LaTeX y sus figuras |
| [`presentacion_final/`](presentacion_final) | Presentación final para la exposición (19 diapositivas): [`presentacion_final.pdf`](presentacion_final/presentacion_final.pdf) y `presentacion_final.pptx`, que trae las notas del orador |
| [`codigo/`](codigo) | Betas 1, 2 y 3 (y la 3b, con `comm.reduce`), validación, ping-pong, barrido de experimentos y análisis. Cada archivo y sus comandos están en [`codigo/README.md`](codigo/README.md) |
| [`codigo/resultados/`](codigo/resultados) | Medición final del 11-nov-2026 (Apple M1 Pro, p = 1 a 8, una sola sesión): tiempos, tablas, figuras, validación y entorno |
| [`docs/`](docs) | Enunciado del proyecto |
| [`archivo/version_luis_2026-11-09/`](archivo/version_luis_2026-11-09) | Versión anterior de este repositorio, sin cambios: código, mediciones en Windows, informe y presentación |

## Uso rápido

Requiere Open MPI (en macOS, `brew install open-mpi`) y Python 3.11. Desde `codigo/`:

```bash
pip install -r requirements.txt
mpirun -np 4 python knn_mpi_beta3.py --validar    # una corrida, validada contra el código del profesor
python analisis.py                                # tablas y figuras a partir de resultados/
caffeinate -i python correr_experimentos.py       # barrido completo en resultados_nuevos/ (unos 35 min)
```

En Windows, las betas también corren con Intel MPI (`pip install impi-rt`, como en el README de `archivo/version_luis_2026-11-09/`). El barrido completo es para macOS o Linux.

## Informe

Desde `informe_final/`, con [Tectonic](https://tectonic-typesetting.github.io): `tectonic informe_final.tex`.
