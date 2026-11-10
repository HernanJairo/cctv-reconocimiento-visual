"""Analítica geométrica: líneas de cruce con histéresis y zonas poligonales.

Todos los puntos de calibración se almacenan NORMALIZADOS [0..1] respecto
del frame de análisis, por lo que son inmunes a cambios de resolución de
la fuente o del tamaño del widget de video.

Cruce de línea: en lugar de comparar solo el eje Y (que falla con líneas
verticales/diagonales y produce dobles conteos por jitter), se calcula el
lado del punto respecto de la recta mediante el signo del producto cruz.
Además se aplica una banda muerta (histéresis) alrededor de la línea: el
lado solo cambia cuando el punto supera `margin_px` de distancia real a la
recta, eliminando conteos repetidos por oscilación del punto-pie.
"""
import math
import threading
from collections import deque

import cv2
import numpy as np


def _side_of_line(p, a, b):
    """Signo del producto cruz: >0 un lado de la recta a->b, <0 el otro."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _dist_to_line(p, a, b):
    num = abs(_side_of_line(p, a, b))
    den = math.hypot(b[0] - a[0], b[1] - a[1])
    return num / den if den > 1e-6 else 0.0


def _chip(img, text, org, color, scale=0.5, thick=1, alpha=0.65):
    """Etiqueta con fondo semitransparente (estilo VMS): el chip no tapa
    la escena y el texto blanco queda legible sobre cualquier fondo."""
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                   scale, thick)
    x, y = org
    x = max(0, min(x, img.shape[1] - tw - 8))
    y = max(th + 8, min(y, img.shape[0] - 2))
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 4, y - th - 6),
                  (x + tw + 4, y + bl + 2), color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thick, cv2.LINE_AA)


def _segments_intersect(p, q, a, b):
    """True si el segmento p->q corta al segmento a->b (test de lados).

    Intersección geométrica pura: p y q deben quedar en lados opuestos de
    la recta a->b Y a y b en lados opuestos de la recta p->q, así el corte
    ocurre dentro del trazo dibujado y no sobre su prolongación.
    """
    s1 = _side_of_line(p, a, b)
    s2 = _side_of_line(q, a, b)
    if (s1 > 0) == (s2 > 0):
        return False                        # p y q del mismo lado de a->b
    s3 = _side_of_line(a, p, q)
    s4 = _side_of_line(b, p, q)
    return (s3 > 0) != (s4 > 0)


class CrossingLine:
    """Línea virtual con detección de cruce con histéresis.

    `kind`: "count" (bidireccional → IN/OUT) o "danger" (cualquier cruce).

    Cada track conserva un historial de sus últimas `HISTORY_LEN`
    posiciones (píxeles del lienzo de análisis). Si el tracker pierde la
    detección algunos frames —o el punto-pie queda dentro de la banda
    muerta— el historial mantiene la trayectoria previa y permite
    reconstruir el lado de la línea desde el que venía el ID, de modo que
    el cruce no se pierde ni se cuenta dos veces.
    """

    HISTORY_LEN = 5                  # posiciones retenidas por track
    PRUNE_EVERY = 300                # cada cuántos updates se podan IDs viejos
    PRUNE_AGE = 600                  # updates sin verse antes de olvidar un ID

    def __init__(self, p1_norm, p2_norm, kind="count", margin_px=5, invert=False):
        self.p1 = p1_norm
        self.p2 = p2_norm
        self.kind = kind
        self.margin_px = margin_px
        self.invert = invert
        self._last_side = {}         # track_id -> último lado fuera de la banda
        self._history = {}           # track_id -> deque de últimas posiciones
        self._seen_at = {}           # track_id -> nº de update (para la poda)
        self._tick = 0

    def _abs(self, pt, w, h):
        return (pt[0] * w, pt[1] * h)

    def _side_at(self, p, a, b):
        """Lado de `p` respecto de la recta: +1/-1, o 0 dentro de la banda."""
        if _dist_to_line(p, a, b) < self.margin_px:
            return 0
        return 1 if _side_of_line(p, a, b) > 0 else -1

    def _prune(self):
        stale = [t for t, s in self._seen_at.items()
                 if self._tick - s > self.PRUNE_AGE]
        for t in stale:
            self._last_side.pop(t, None)
            self._history.pop(t, None)
            self._seen_at.pop(t, None)

    def reset(self):
        self._last_side.clear()
        self._history.clear()
        self._seen_at.clear()

    def forget_track(self, track_id):
        self._last_side.pop(track_id, None)
        self._history.pop(track_id, None)
        self._seen_at.pop(track_id, None)

    def update(self, track_id, foot_px, frame_w, frame_h):
        """Devuelve "IN" | "OUT" | "CROSS" | None según el tipo de línea."""
        a = self._abs(self.p1, frame_w, frame_h)
        b = self._abs(self.p2, frame_w, frame_h)

        # Trayectoria: siempre se registra la posición, incluso dentro de
        # la banda, para conservar el vector de movimiento entre frames
        # (incluidos huecos de detección que el tracker reasocia después).
        self._tick += 1
        self._seen_at[track_id] = self._tick
        hist = self._history.setdefault(track_id,
                                        deque(maxlen=self.HISTORY_LEN))
        prev_pos = hist[-1] if hist else None
        hist.append(foot_px)
        if self._tick % self.PRUNE_EVERY == 0:
            self._prune()

        # Intersección segmento-segmento pura entre el desplazamiento
        # prev_pos->foot y el trazo a->b: detecta el cruce aunque el pie
        # aterrice dentro de la banda muerta o el track desaparezca justo
        # después (zancadas rápidas, línea "saltada" en 1 solo cuadro).
        seg_cross = (prev_pos is not None
                     and _segments_intersect(prev_pos, foot_px, a, b))

        side = self._side_at(foot_px, a, b)

        # Banda muerta: dentro del margen no actualizamos el lado conocido,
        # así una oscilación del pie sobre la línea nunca cuenta dos veces.
        # Excepción: si el segmento de movimiento cruzó el trazo, el cruce
        # es real y el lado nuevo se resuelve por el signo crudo del
        # producto cruz (la banda no puede anular un cruce geométrico).
        if side == 0:
            if not seg_cross:
                return None
            side = 1 if _side_of_line(foot_px, a, b) > 0 else -1

        prev = self._last_side.get(track_id)
        if prev is None:
            if seg_cross:
                # Track nuevo o reasignado cuyo primer salto ya cruzó: el
                # lado previo es el signo crudo de la posición anterior.
                prev = 1 if _side_of_line(prev_pos, a, b) > 0 else -1
            else:
                # Sin lado previo registrado (track nuevo o reasignado tras
                # un hueco de detección): se reconstruye desde la
                # trayectoria, buscando la última posición retenida fuera
                # de la banda.
                for p in reversed(list(hist)[:-1]):
                    prev = self._side_at(p, a, b)
                    if prev != 0:
                        break
                else:
                    prev = 0
        self._last_side[track_id] = side

        if prev in (0, side):
            return None

        if self.kind == "danger":
            return "CROSS"
        direction = "IN" if side > 0 else "OUT"
        if self.invert:
            direction = "OUT" if direction == "IN" else "IN"
        return direction

    def draw(self, img, w, h, color, label):
        a = tuple(map(int, self._abs(self.p1, w, h)))
        b = tuple(map(int, self._abs(self.p2, w, h)))
        cv2.line(img, a, b, color, 2, cv2.LINE_AA)
        cv2.circle(img, a, 4, color, -1, cv2.LINE_AA)
        cv2.circle(img, b, 4, color, -1, cv2.LINE_AA)
        _chip(img, label, (a[0], max(a[1] - 10, 16)), color, 0.5)


class RestrictedZone:
    """Polígono restringido (n puntos) evaluado con pointPolygonTest."""

    def __init__(self, points_norm):
        self.points = list(points_norm)

    def _abs_pts(self, w, h):
        return np.array([(p[0] * w, p[1] * h) for p in self.points],
                        dtype=np.int32).reshape((-1, 1, 2))

    def contains(self, foot_px, frame_w, frame_h):
        pts = self._abs_pts(frame_w, frame_h)
        return cv2.pointPolygonTest(pts, (float(foot_px[0]), float(foot_px[1])), False) >= 0

    def draw(self, img, w, h, color=(0, 0, 255)):
        pts = self._abs_pts(w, h)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
        cv2.polylines(img, [pts], isClosed=True, color=color,
                      thickness=2, lineType=cv2.LINE_AA)
        p0 = self._abs(self.points[0], w, h)
        _chip(img, "ZONA RESTRINGIDA",
              (int(p0[0]), max(int(p0[1]) - 12, 18)), color, 0.6)

    @staticmethod
    def _abs(pt, w, h):
        return (pt[0] * w, pt[1] * h)


class Calibration:
    """Estado de calibración compartido GUI <-> motor (thread-safe)."""

    MODES = ("ninguno", "linea_conteo", "linea_alarma", "zona")

    def __init__(self):
        self._lock = threading.Lock()
        self.mode = "ninguno"
        self.count_line = []     # hasta 2 puntos normalizados
        self.danger_line = []
        self.zone = []           # hasta 4 puntos normalizados
        self.revision = 0        # aumenta con cada cambio (para el motor)

    def set_mode(self, mode):
        with self._lock:
            self.mode = mode
            if mode == "linea_conteo":
                self.count_line.clear()
            elif mode == "linea_alarma":
                self.danger_line.clear()
            elif mode == "zona":
                self.zone.clear()
            self.revision += 1

    def add_point(self, xn, yn):
        """Agrega un punto normalizado al modo activo. True si completó."""
        with self._lock:
            target = {"linea_conteo": (self.count_line, 2),
                      "linea_alarma": (self.danger_line, 2),
                      "zona": (self.zone, 4)}.get(self.mode)
            if target is None:
                return False
            pts, needed = target
            pts.append((xn, yn))
            self.revision += 1
            if len(pts) >= needed:
                self.mode = "ninguno"
                self.revision += 1
                return True
            return False

    def clear(self):
        with self._lock:
            self.count_line.clear()
            self.danger_line.clear()
            self.zone.clear()
            self.mode = "ninguno"
            self.revision += 1

    def snapshot(self):
        """Copia inmutable del estado para uso del motor."""
        with self._lock:
            return {
                "mode": self.mode,
                "count_line": tuple(self.count_line),
                "danger_line": tuple(self.danger_line),
                "zone": tuple(self.zone),
                "revision": self.revision,
            }
