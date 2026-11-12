#!/bin/bash
# Proyecto CS4052 - Caso 5 KNN - Grupo 7
# Envoltorio: el barrido lo hace correr_experimentos.py (ver README.md). Por ejemplo:
#     caffeinate -i ./correr_experimentos.sh                (= python correr_experimentos.py)
#     PY=/ruta/a/python ./correr_experimentos.sh prueba --rapido
exec "${PY:-python}" "$(dirname "$0")/correr_experimentos.py" "$@"
