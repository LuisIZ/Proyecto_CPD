# KNN en paralelo con MPI

Proyecto de laboratorio de CS4052 Computación Paralela y Distribuida (UTEC, 2026-II), Caso 5, Grupo 7. Paraleliza con mpi4py la clasificación de dígitos de `load_digits`.

## Instalación

Con Python 3.14 en Windows, en una carpeta fuera de OneDrive:

```bash
python -m venv C:\Users\<usuario>\.venvs\proyecto-cpd
C:\Users\<usuario>\.venvs\proyecto-cpd\Scripts\python -m pip install -r requirements.txt
```

`requirements.txt` instala Intel MPI en Windows mediante `impi-rt`. En otros sistemas hace falta tener MPI instalado.

## Ejecución

Todos los comandos se ejecutan desde `src/`, con el Python del entorno virtual.

Una versión con 4 procesos (en Windows, `mpiexec` está en `Library\bin` del entorno virtual, y `I_MPI_ROOT` debe apuntar a `Library`):

```bash
mpiexec -n 4 python beta2_knn_mpi.py --n 1797
```

Comparar las tres versiones con el código base. No necesita configurar `mpiexec`:

```bash
python validar_betas.py
```

Ejecutar la malla de experimentos y generar las gráficas:

```bash
python experimentos.py --repeticiones 5 --salida ../results/final
python graficas.py --entrada ../results/final
```

`experimentos.py` continúa donde quedó si se interrumpe. Conviene ejecutarlo con la máquina libre y conectada a la corriente.

## Versiones

| Archivo | Contenido |
|---|---|
| `drive/knn_digits_sec.py` | Código secuencial base, sin modificar |
| `src/beta1_knn_mpi.py` | Beta 1: `bcast`, `scatter` y `gather` con bloques desiguales |
| `src/beta2_knn_mpi.py` | Beta 2: medición de tiempos y FLOP, sin barreras |
| `src/beta3_knn_mpi.py` | Beta 3: paso 6 optimizado con `comm.reduce` |
