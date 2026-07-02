"""
validation/ — Framework de Validación Pre-Registrada (Nodo-51 F5)

Contiene hipótesis pre-registradas congeladas. Los umbrales NO se modifican
hasta alcanzar n_stop. Modificar antes = p-hacking.

Regla fundamental (MM-5, Nodo-51):
  Una cohorte (n grande, calibrada) derrota a un case report (n=2, 100% hit).
  Un hit% espectacular en n pequeño genera una HIPÓTESIS PRE-REGISTRADA,
  no una calibración.
"""
