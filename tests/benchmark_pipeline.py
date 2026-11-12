"""Benchmark de estrés del pipeline captura -> inferencia -> slot de salida.

Un productor sintético emula un stream IP a FPS fijo sobre el FrameSlot
real (latest-frame-wins); el InferWorker corre con el modelo y los
umbrales reales de settings.yaml, y un consumidor drena el slot de
salida durante N segundos. Se mide:

  * FPS de "captura" (producidos) vs FPS de inferencia (entregados)
  * Latencia extremo a extremo cuadro a cuadro (productor -> anotado)
  * Deriva de latencia (acumulacion de retardo: debe ser plana)
  * Estabilidad de RAM (RSS del proceso, muestreado 1 vez/seg)

Uso:
    python tests/benchmark_pipeline.py                  # 60 s a 30 fps
    python tests/benchmark_pipeline.py --seconds 30 --fps 30
    python tests/benchmark_pipeline.py --selftest       # solo chequeos
"""
import argparse
import os
import queue
import sys
import threading
import time

import numpy as np
import psutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.capture import FrameSlot, read_freshest          # noqa: E402
from src.config import load_settings                      # noqa: E402
from src.geometry import Calibration                      # noqa: E402
from src.paths import VIDEOS_DIR                          # noqa: E402


# ---------- estampa de latencia (idx embebido en pixeles) ----------

def _stamp(frame, idx):
    """Escribe idx (uint32 LE) en dos esquinas. El productor emite ya al
    tamano del lienzo de analisis, asi el resize del engine no lo mueve."""
    for k in range(4):
        b = (idx >> (8 * k)) & 0xFF
        frame[0, k] = b
        frame[-1, -(k + 1)] = b


def _unstamp(frame, max_idx):
    """Decodifica el idx estampado; elige la esquina valida (una caja
    dibujada sobre la otra esquina puede haberla pisado)."""
    cands = (
        int.from_bytes(bytes(int(v) for v in frame[0, :4, 0]), "little"),
        int.from_bytes(bytes(int(v) for v in frame[-1, -4:, 0][::-1]),
                       "little"),
    )
    for c in cands:
        if 0 <= c <= max_idx:
            return c
    return -1


# ---------- productor sintetico (emula stream IP) ----------

class _Producer(threading.Thread):
    def __init__(self, slot, settings, fps, seconds):
        super().__init__(daemon=True, name="BenchProducer")
        self.slot = slot
        self.s = settings
        self.fps = fps
        self.seconds = seconds
        self.ts = []                        # ts[i] = instante de entrega
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _frames(self):
        """Frames de prueba.mp4 en ciclo; si falta, escena sintetica."""
        import cv2
        aw, ah = self.s.analysis_width, self.s.analysis_height
        path = os.path.join(VIDEOS_DIR, "prueba.mp4")
        cap = cv2.VideoCapture(path) if os.path.exists(path) else None
        if cap is not None and not cap.isOpened():
            cap = None
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, (ah, aw, 3), dtype=np.uint8)
        x = 0
        try:
            while not self._stop.is_set():
                if cap is not None:
                    ok, f = cap.read()
                    if not ok:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    yield cv2.resize(f, (aw, ah))
                else:
                    f = base.copy()
                    f[100:400, x:x + 80] = 200    # rectangulo en movimiento
                    x = (x + 12) % (aw - 80)
                    yield f
        finally:
            if cap is not None:
                cap.release()

    def run(self):
        self.ts0 = time.monotonic()
        period = 1.0 / self.fps
        nxt = time.monotonic()
        for idx, frame in enumerate(self._frames()):
            if self._stop.is_set() or time.monotonic() - self.ts0 > self.seconds:
                break
            _stamp(frame, idx)
            self.ts.append(time.monotonic())    # ts ANTES de set()
            self.slot.set(frame)
            nxt += period
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.monotonic()          # re-anclar: no acumular


class _NullWriter:
    """Mismo API que EventWriter pero sin I/O (no ensucia seguridad.db)."""
    def enqueue_event(self, *a):
        pass

    def enqueue_alert(self, *a):
        pass

    def stop(self):
        pass


# ---------- selftest del vaciado de backlog ----------

class _FakeCap:
    """Emula un VideoCapture con `backlog` frames encolados; cuando la
    cola se vacia, cada grab() 'espera a la red' period_s segundos."""
    def __init__(self, backlog, period_s):
        self._backlog = backlog
        self._period = period_s
        self.grabs = 0
        self._last = None

    def grab(self):
        self.grabs += 1
        if self._backlog > 0:
            self._backlog -= 1
            self._last = f"stale_{self.grabs}"
            return True
        time.sleep(self._period)
        self._last = f"fresh_{self.grabs}"
        return True

    def retrieve(self):
        return True, self._last


def selftest():
    # Caso 1: 5 frames encolados + frescos cada 30 ms -> drena los 5
    # viejos y devuelve el fresco (grab bloqueante detectado).
    cap = _FakeCap(backlog=5, period_s=0.030)
    ok, frame = read_freshest(cap, drain_max=30, fresh_s=0.020)
    assert ok and frame == "fresh_6", (ok, frame, cap.grabs)
    assert cap.grabs == 6, cap.grabs
    print("selftest 1: backlog de 5 drenado, entrega el frame fresco")

    # Caso 2: backlog enorme (100) -> vaciado acotado a drain_max, nunca
    # se bloquea; entrega el mas reciente disponible.
    cap = _FakeCap(backlog=100, period_s=0.030)
    ok, frame = read_freshest(cap, drain_max=30, fresh_s=0.020)
    assert ok and frame == "stale_30", (ok, frame)
    assert cap.grabs == 30, cap.grabs
    print("selftest 2: vaciado acotado a drain_max=30 (sin bloqueo)")

    # Caso 3: FrameSlot latest-frame-wins.
    slot = FrameSlot()
    slot.set("v1")
    slot.set("v2")
    f, _ = slot.latest()
    assert f == "v2"
    print("selftest 3: FrameSlot devuelve siempre el ultimo frame")
    print("SELFTEST OK")


# ---------- benchmark ----------

def _pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def run(seconds, fps):
    s = load_settings()

    # Regla de arquitectura: el modelo se carga + warm-up en el hilo
    # principal, ANTES de arrancar el productor (FFmpeg/OpenMP).
    from src.engine import load_model, InferWorker
    model = load_model(s)

    in_slot, out_slot = FrameSlot(), FrameSlot()
    status_q = queue.Queue(maxsize=512)
    worker = InferWorker(s, in_slot, out_slot, Calibration(),
                         _NullWriter(), status_q, model=model)
    prod = _Producer(in_slot, s, fps=fps, seconds=seconds)

    worker.start()
    prod.ts0 = time.monotonic()
    prod.start()

    proc = psutil.Process()
    proc.memory_info()                      # primer RSS (warm del contador)
    lat_ms, rss = [], []
    delivered = undecodable = 0
    last_seq = -1
    t_start = time.monotonic()
    t_end = t_start + seconds

    while time.monotonic() < t_end:
        data, last_seq = out_slot.wait_new(last_seq, timeout=0.25)
        now = time.monotonic()
        if not rss or now - rss[-1][0] >= 1.0:
            rss.append((now, proc.memory_info().rss / 1e6))
        if data is None:
            continue
        delivered += 1
        idx = _unstamp(data["frame"], len(prod.ts) - 1)
        if 0 <= idx < len(prod.ts):
            lat_ms.append((now - prod.ts[idx]) * 1000.0)
        else:
            undecodable += 1

    elapsed = time.monotonic() - t_start
    prod.stop()
    worker.stop()
    worker.join(timeout=5.0)

    produced = len(prod.ts)
    cap_fps = produced / elapsed
    inf_fps = delivered / elapsed
    p50, p95, pmax = _pct(lat_ms, .50), _pct(lat_ms, .95), _pct(lat_ms, 1.0)

    # Deriva de latencia: media del primer 10% vs ultimo 10% de muestras.
    w = max(1, len(lat_ms) // 10)
    drift = 0.0
    if len(lat_ms) >= 2 * w:
        a, b = np.mean(lat_ms[:w]), np.mean(lat_ms[-w:])
        drift = (b - a) / a * 100.0 if a else 0.0

    rss_v = [v for _, v in rss]
    rss_first = np.mean(rss_v[:3]) if len(rss_v) >= 3 else (rss_v[0] if rss_v else 0)
    rss_last = np.mean(rss_v[-3:]) if rss_v else 0
    rss_drift = rss_last - rss_first

    rss_peak = max(rss_v) if rss_v else 0
    print(f"""
=== RESULTADOS ({elapsed:.1f} s de carga sostenida) ===================
Modelo          : {s.model_file} imgsz={s.imgsz} conf={s.conf} iou={s.iou}
Producidos      : {produced:5d} frames  ({cap_fps:5.1f} FPS captura)
Entregados      : {delivered:5d} frames  ({inf_fps:5.1f} FPS inferencia)
Descartados     : {produced - delivered:5d} (latest-frame-wins; normal si
                  la inferencia es mas lenta que la captura)
Latencia e2e    : p50 {p50:6.1f} ms | p95 {p95:6.1f} ms | max {pmax:6.1f} ms
                  (muestras: {len(lat_ms)}, indeco.: {undecodable})
Deriva latencia : {drift:+6.1f} %  (inicio vs fin; ~0 = SIN acumulacion)
RAM (RSS)       : {rss_first:6.0f} MB -> {rss_last:6.0f} MB \
                  (deriva {rss_drift:+.0f} MB, pico {rss_peak:.0f} MB)
=====================================================================""")

    ok = True
    if p95 < 200:
        print("PASS  p95 de latencia < 200 ms")
    else:
        ok = False
        print("WARN  p95 > 200 ms: la inferencia limita; la latencia sigue"
              " acotada si la deriva es plana. Opciones: yolov8n/ONNX o"
              " imgsz 512.")
    if drift < 20:
        # Solo la deriva POSITIVA grande indica retardo acumulativo; una
        # deriva negativa es el warm-up estabilizando la medición.
        print("PASS  deriva de latencia plana o descendente (sin delay acumulativo)")
    else:
        ok = False
        print("FAIL  la latencia CRECE con el tiempo -> retardo acumulativo")
    if rss_drift < 100:
        print("PASS  RAM estable (deriva < 100 MB en la sesion)")
    else:
        ok = False
        print("FAIL  deriva de RAM significativa -> posible fuga")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return 0

    selftest()
    return run(args.seconds, args.fps)


if __name__ == "__main__":
    sys.exit(main())
