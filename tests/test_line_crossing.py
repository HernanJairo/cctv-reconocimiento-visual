"""Test: cruce de línea anti-salto (intersección segmento-segmento).

Simula trayectorias del punto-pie sobre líneas de conteo e intrusión para
verificar que el conteo se dispara INMEDIATAMENTE incluso cuando el track
"salta" la línea en un solo cuadro —el escenario típico de frame drops
en streams IP— y que no hay falsos conteos por jitter dentro de la banda.

Correr:  python tests/test_line_crossing.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry import CrossingLine, _segments_intersect

W, H = 960, 540
# Línea de conteo horizontal normalizada -> a=(96,270) b=(864,270) en px.
# side(p) = 768*(y-270):  y<270 -> lado -1 ("OUT"),  y>270 -> lado +1 ("IN")
LINE = ((0.1, 0.5), (0.9, 0.5))


def line(kind="count"):
    return CrossingLine(*LINE, kind=kind, margin_px=5)


# --- 0) Primitiva geométrica ---
assert _segments_intersect((480, 200), (480, 400), (96, 270), (864, 270))
assert not _segments_intersect((100, 200), (700, 100), (96, 270), (864, 270))
# la prolongación NO cuenta: el salto cruza la recta pero fuera del trazo
assert not _segments_intersect((20, 200), (20, 400), (96, 270), (864, 270))
print("0) _segments_intersect: corta dentro del trazo, no en la prolongación")

# --- 1) Cruce lento (varios cuadros): un solo conteo, dirección IN ---
ln = line()
res = [ln.update(1, (480, y), W, H) for y in range(250, 290, 4)]
assert [r for r in res if r] == ["IN"], res
print("1) Cruce lento: exactamente 1 conteo, dirección correcta")

# --- 2) Salto de 1 solo cuadro (objeto veloz / frame drop): cuenta igual ---
ln = line()
assert ln.update(2, (480, 200), W, H) is None        # lado -1
assert ln.update(2, (480, 400), W, H) == "IN"        # teletransporte -> cuenta
print("2) Salto 200px en 1 cuadro: conteo registrado")

# --- 3) El salto ATERRIZA dentro de la banda muerta: seg_cross lo rescata ---
ln = line()
assert ln.update(3, (480, 200), W, H) is None
# y=272 queda a 2px de la línea (< margin 5): sin intersección de
# segmentos esto quedaría "en banda" y el cruce se perdería.
assert ln.update(3, (480, 272), W, H) == "IN"
print("3) Salto que aterriza en la banda muerta: conteo igual (anti-salto)")

# --- 4) Movimiento dentro de la banda SIN cruzar: no cuenta ---
ln = line()
assert ln.update(4, (480, 200), W, H) is None        # lado -1
assert ln.update(4, (480, 266), W, H) is None        # banda, mismo lado
assert ln.update(4, (480, 268), W, H) is None        # jitter en banda
print("4) Jitter dentro de la banda del mismo lado: 0 conteos")

# --- 5) Teletransporte dentro del mismo lado: NO cuenta ---
ln = line()
assert ln.update(5, (100, 200), W, H) is None
assert ln.update(5, (700, 100), W, H) is None
print("5) Salto grande sin tocar la línea: 0 conteos")

# --- 6) Track perdido y reasociado ya cruzado (hueco de detección) ---
# El historial conserva la última posición conocida como prev_pos, así
# el segmento [última vista -> reaparición] se evalúa aunque hayan
# pasado N cuadros sin detección del ID.
ln = line()
assert ln.update(6, (480, 200), W, H) is None        # última vista: lado -1
# ... tracker pierde el ID durante varios frames ...
assert ln.update(6, (480, 500), W, H) == "IN"        # reaparece cruzado
print("6) Track con hueco de detección: el cruce se reconstruye y cuenta")

# --- 7) Trayectoria que pasa EXACTAMENTE por un extremo del trazo ---
# Salto vertical por x=96: cruza el segmento justo en su punta a=(96,270)
# (producto cruz == 0). Sin el chequeo on-segment este cruce se perdía.
ln = line()
assert ln.update(7, (96, 100), W, H) is None
assert ln.update(7, (96, 272), W, H) == "IN"         # aterriza en banda
print("7) Salto que pasa por el extremo exacto de la línea: contado")

# --- 8) Línea de intrusión (danger): cualquier cruce -> "CROSS" ---
ln = line(kind="danger")
assert ln.update(8, (480, 200), W, H) is None
assert ln.update(8, (480, 400), W, H) == "CROSS"
print("8) Línea prohibida: salto en 1 cuadro dispara CROSS")

# --- 9) Línea vertical: la intersección no depende de la orientación ---
lv = CrossingLine((0.5, 0.1), (0.5, 0.9), kind="count", margin_px=5)
assert lv.update(9, (300, 270), W, H) is None        # lado +1
assert lv.update(9, (660, 270), W, H) == "OUT"       # salto 360px -> -1
print("9) Línea vertical: salto horizontal en 1 cuadro contado")

# --- 10) Tras cruzar, no hay doble conteo al quedarse del otro lado ---
ln = line()
assert ln.update(10, (480, 200), W, H) is None
assert ln.update(10, (480, 400), W, H) == "IN"
assert ln.update(10, (480, 420), W, H) is None       # sigue del mismo lado
assert ln.update(10, (480, 350), W, H) is None       # retrocede sin cruzar
print("10) Sin doble conteo permaneciendo del lado destino")

print("\nTODOS LOS TESTS DE CRUCE ANTI-SALTO PASARON")
