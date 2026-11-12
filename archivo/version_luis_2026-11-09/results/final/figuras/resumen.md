# Resumen de mediciones

Ping-pong: a = 3.602e-05 s, b = 5.846e-09 s por dato float64.

Código base, n = 1797: mediana 3.4180 s (5 repeticiones).

## T_s frente a p = 1

| n | T_s (s) | gather p = 1 | reducción p = 1 |
|---|---|---|---|
| 450 | 0.0103 | 0.0112 | 0.0107 |
| 900 | 0.0390 | 0.0400 | 0.0391 |
| 1350 | 0.0931 | 0.0955 | 0.0958 |
| 1797 | 0.1662 | 0.1682 | 0.1675 |
| 3600 | 0.7259 | 0.7370 | 0.7262 |
| 7200 | 2.9441 | 2.9837 | 2.9809 |

## Con gather (Beta 2)
Coeficientes (s): c1 = 5.179e-08, c2 = 1.124e-06, c3 = 1.020e-07, c4 = 4.011e-04
Error relativo medio del ajuste: 8.1 %
Ajuste de T_comp: [5.236e-08 8.175e-07], error 10.2 %
Ajuste de T_comm (p > 1): [9.007e-07 3.496e-05 1.849e-07], error 28.3 %

| n | p* estimado | cruce comp/comm (inf: no se cruzan) | mejor p medido | speedup máximo |
|---|---|---|---|---|
| 450 | 4.6 | inf | 4 | 2.04 |
| 900 | 6.4 | inf | 4 | 2.81 |
| 1350 | 7.9 | inf | 4 | 3.47 |
| 1797 | 9.1 | inf | 8 | 3.96 |
| 3600 | 12.9 | inf | 16 | 6.33 |
| 7200 | 18.2 | inf | 16 | 8.00 |

| Condición | (n, p) | E medida | E del modelo |
|---|---|---|---|
| $n \propto p^2$ | (450, 4) | 0.511 | 0.496 |
| $n \propto p^2$ | (1797, 8) | 0.495 | 0.545 |
| $n \propto p^2$ | (7200, 16) | 0.500 | 0.558 |
| $n \propto p \log_2 p$ | (450, 4) | 0.511 | 0.496 |
| $n \propto p \log_2 p$ | (1350, 8) | 0.411 | 0.469 |
| $n \propto p \log_2 p$ | (3600, 16) | 0.396 | 0.385 |

## Con reducción (Beta 3)
Coeficientes (s): c1 = 4.870e-08, c3 = 3.945e-06, c4 = 1.951e-04
Error relativo medio del ajuste: 11.5 %
Ajuste de T_comp: [4.872e-08 3.000e-06], error 10.7 %
Ajuste de T_comm (p > 1): [1.231e-06 0.000e+00], error 34.9 %

| n | p* estimado | cruce comp/comm (inf: no se cruzan) | mejor p medido | speedup máximo |
|---|---|---|---|---|
| 450 | 3.5 | inf | 4 | 2.00 |
| 900 | 7.3 | inf | 4 | 2.86 |
| 1350 | 11.1 | inf | 16 | 3.60 |
| 1797 | 15.0 | inf | 16 | 4.28 |
| 3600 | 30.4 | inf | 16 | 6.90 |
| 7200 | 61.2 | inf | 16 | 7.77 |

| Condición | (n, p) | E medida | E del modelo |
|---|---|---|---|
| $n \propto p^2$ | (450, 4) | 0.499 | 0.385 |
| $n \propto p^2$ | (1797, 8) | 0.489 | 0.474 |
| $n \propto p^2$ | (7200, 16) | 0.486 | 0.580 |
| $n \propto p \log_2 p$ | (450, 4) | 0.499 | 0.385 |
| $n \propto p \log_2 p$ | (1350, 8) | 0.413 | 0.401 |
| $n \propto p \log_2 p$ | (3600, 16) | 0.431 | 0.407 |

## Precisión

| n | accuracy |
|---|---|
| 450 | 0.9556 |
| 900 | 0.9667 |
| 1350 | 0.9889 |
| 1797 | 0.9833 |
