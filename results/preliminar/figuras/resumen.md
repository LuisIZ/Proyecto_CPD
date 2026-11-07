# Resumen de mediciones

Código base, n = 1797: mediana 7.0175 s (3 repeticiones).

## Con gather (Beta 2)
Coeficientes (s): c1 = 6.234e-08, c2 = 1.282e-06, c3 = 0.000e+00, c4 = 8.140e-04
Error relativo medio del ajuste: 7.5 %

| n | p* estimado | mejor p medido | speedup máximo | accuracy |
|---|---|---|---|---|
| 450 | 4.7 | 4 | 1.64 | 0.9556 |
| 900 | 6.6 | 8 | 2.88 | 0.9778 |
| 1350 | 8.1 | 8 | 4.47 | 0.9704 |
| 1797 | 9.3 | 8 | 5.43 | 0.9833 |
| 3600 | 13.2 | 16 | 6.58 | 0.9931 |
| 7200 | 18.7 | 16 | 8.22 | 1.0000 |

Isoeficiencia (n, p, E): (450, 4, 0.409), (1797, 8, 0.678), (7200, 16, 0.514)

## Con reducción (Beta 3)
Coeficientes (s): c1 = 6.026e-08, c3 = 4.872e-06, c4 = 1.216e-04
Error relativo medio del ajuste: 9.5 %

| n | p* estimado | mejor p medido | speedup máximo | accuracy |
|---|---|---|---|---|
| 450 | 3.7 | 8 | 2.11 | 0.9556 |
| 900 | 7.5 | 8 | 2.58 | 0.9778 |
| 1350 | 11.4 | 8 | 4.30 | 0.9704 |
| 1797 | 15.2 | 8 | 4.57 | 0.9833 |
| 3600 | 30.7 | 16 | 6.23 | 0.9931 |
| 7200 | 61.5 | 16 | 8.32 | 1.0000 |

Isoeficiencia (n, p, E): (450, 4, 0.482), (1350, 8, 0.537), (3600, 16, 0.389)
