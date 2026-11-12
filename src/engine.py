"""Motor de visión: InferWorker.

Consume el frame MÁS RECIENTE del FrameSlot de captura (los intermedios se
descartan → latencia acotada), ejecuta YOLO+ByteTrack sobre un lienzo de
análisis de resolución fija, aplica la analítica (conteo/alarmas/zonas),
dibuja overlays y publica el frame anotado + estadísticas para la GUI.

Nunca toca Tkinter ni hace I/O de disco: la evidencia y los INSERT se
encolan al EventWriter y los mensajes a status_q.
"""
import gc
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

# IMPORTANTE: torch y ultralytics se importan aquí (hilo principal, en el
# arranque) y NO dentro del worker. Importar torch o crear el modelo en un
# hilo secundario puede bloquearse en Windows (conflicto de runtimes
# OpenMP/MKL durante la carga perezosa de DLLs y chequeos del SDK).
import torch
from ultralytics import YOLO

from .capture import FrameSlot
from .geometry import CrossingLine, RestrictedZone
from .storage import new_capture_path


def _download_model(path, emit=print):
    """Descarga el .pt faltante dentro de models/ con el downloader de
    Ultralytics (misma fuente y verificación que usa YOLO internamente).

    Solo aplica a pesos oficiales (yolov8*.pt, yolo11*.pt, …). Para
    cualquier otro archivo informa cómo resolverlo a mano.
    """
    name = os.path.basename(path)
    emit(f"{name} no encontrado; descargando a models/…")
    try:
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(path)
    except Exception as e:
        raise FileNotFoundError(
            f"No se pudo descargar el modelo '{name}': {e}") from e
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Modelo no encontrado: {path}. Descárguelo manualmente a "
            f"'models/' o corrija 'model_file' en config/settings.yaml.")


def load_model(settings, emit=print):
    """Carga el modelo YOLO y hace warm-up de la 1ra inferencia.

    DEBE llamarse desde el hilo principal antes de iniciar cualquier
    VideoCapture/FFmpeg: la primera inferencia inicializa OpenMP/MKL de
    forma perezosa y puede bloquearse si otra DLL nativa está activa.
    """
    torch.set_num_threads(settings.torch_threads)
    path = settings.model_path
    if not os.path.exists(path):
        _download_model(path, emit)
    emit(f"Cargando modelo {os.path.basename(path)}…")
    model = YOLO(path)
    emit("Warm-up de inferencia…")
    model.track(
        np.zeros((settings.analysis_height, settings.analysis_width, 3),
                 dtype=np.uint8),
        persist=True, classes=settings.detect_classes, conf=settings.conf,
        iou=settings.iou, imgsz=settings.imgsz, tracker=settings.tracker_cfg,
        verbose=False)
    emit("Modelo listo.")
    return model

# Paleta semántica VMS (OpenCV usa BGR):
COLOR_COUNT = (212, 180, 6)      # línea de conteo: cian   #06b6d4
COLOR_DANGER = (11, 158, 245)    # línea prohibida: ámbar #f59e0b
COLOR_ZONE = (94, 63, 244)       # polígono: carmesí       #f43f5e
COLOR_OK = (129, 185, 16)        # track normal: esmeralda #10b981
COLOR_BAD = (94, 63, 244)        # intrusión: carmesí      #f43f5e
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _tag(img, text, org, color, scale=0.5, thick=1, alpha=0.65):
    """Etiqueta con chip semitransparente: legible sin tapar la imagen.

    El blend se aplica SOLO al ROI del chip (no al frame completo): con
    varias detecciones por frame, copiar+mezclar 1.7 MB por etiqueta era
    el segundo mayor costo por frame tras la propia inferencia.
    """
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org
    x = max(0, min(x, img.shape[1] - tw - 6))
    y = max(th + 8, min(y, img.shape[0] - 2))
    x0, y0 = max(0, x - 4), max(0, y - th - 6)
    x1, y1 = min(img.shape[1], x + tw + 4), min(img.shape[0], y + bl + 2)
    roi = img[y0:y1, x0:x1]
    chip = np.full(roi.shape, color, dtype=np.uint8)
    cv2.addWeighted(chip, alpha, roi, 1 - alpha, 0, roi)
    cv2.putText(img, text, (x, y), FONT, scale,
                (255, 255, 255), thick, cv2.LINE_AA)


class InferWorker(threading.Thread):
    def __init__(self, settings, in_slot, out_slot, calibration, writer,
                 status_q, model=None):
        super().__init__(daemon=True, name="InferWorker")
        self.s = settings
        self._model = model
        self.in_slot = in_slot
        self.out_slot = out_slot
        self.calibration = calibration
        self.writer = writer
        self.status_q = status_q
        self.ctl = queue.Queue()

        self.total_in = 0
        self.total_out = 0
        self.aforo = 0
        self._alert_cooldown = {}      # track_id -> último timestamp de alerta
        self._last_alert = None        # (track_id, ts) para el banner
        self._calib_rev = -1
        self._count_line = None
        self._danger_line = None
        self._zone = None
        self._fps_ema = 0.0
        # imgsz efectivo: la GUI lo ajusta por tipo de fuente (set_imgsz)
        # para que un stream IP infiera con menos carga que webcam/archivo.
        self._imgsz = settings.imgsz
        # --- inferencia ASÍNCRONA ---
        # Un hilo dedicado corre model.track() de forma continua sobre el
        # frame más reciente que le ofrece el worker (1 de cada
        # `infer_stride` publicados). El worker publica SIEMPRE a la
        # cadencia de la fuente (30-60 FPS de visor) redibujando las
        # últimas cajas conocidas; las cajas se refrescan cada ~T_inf ms.
        self._infer_tick = 0
        self._last_boxes = np.empty((0, 4))
        self._last_ids = np.empty(0, dtype=int)
        self._last_bad = np.empty(0, dtype=bool)  # flag intruso por caja
        self._pub_times = deque(maxlen=32)  # ventana de publicaciones (FPS real)
        self._infer_slot = FrameSlot()      # worker -> hilo de inferencia
        self._infer_ctl = queue.Queue()     # comandos hacia el hilo
        self._infer_stop = threading.Event()
        self._infer_thread = None
        self._det_times = deque(maxlen=16)  # ventana de inferencias (Hz YOLO)
        self._det_fps = 0.0
        # Objetos de analítica propios del hilo de inferencia (estado de
        # tracks por línea/zona — el worker dibuja los SUYOS en paralelo).
        self._an_rev = -1
        self._an_count = None
        self._an_danger = None
        self._an_zone = None
        # Anti-flicker: IDs detectados en la inferencia anterior pero
        # ausentes en la actual se dibujan 1 ciclo más (detecciones
        # lejanas con score variable parpadean menos).
        self._carry = {}
        self._gc_tick = 0                   # GC periódico (RAM limpia)

    # --- API (GUI) ---
    def send(self, *cmd):
        self.ctl.put(cmd)

    def stop(self):
        self.ctl.put(("stop",))

    def _emit(self, kind, **kw):
        try:
            self.status_q.put_nowait({"src": "engine", "kind": kind, **kw})
        except queue.Full:
            pass

    def _drain_controls(self):
        while True:
            try:
                cmd = self.ctl.get_nowait()
            except queue.Empty:
                return False
            if cmd[0] == "stop":
                return True
            if cmd[0] == "reset_counters":
                self.total_in = self.total_out = 0
                self._alert_cooldown.clear()
                if self._count_line:
                    self._count_line.reset()
                if self._danger_line:
                    self._danger_line.reset()
                self._infer_ctl.put(("reset_counters",))
            elif cmd[0] == "set_imgsz":
                self._imgsz = int(cmd[1])
                self._infer_ctl.put(("set_imgsz", int(cmd[1])))

    def _sync_calibration(self, snap):
        if snap["revision"] == self._calib_rev:
            return snap["mode"], snap
        self._calib_rev = snap["revision"]
        self._count_line = (CrossingLine(*snap["count_line"], kind="count",
                                       margin_px=self.s.line_hysteresis_px,
                                       invert=self.s.invert_direction)
                            if len(snap["count_line"]) == 2 else None)
        self._danger_line = (CrossingLine(*snap["danger_line"], kind="danger",
                                        margin_px=self.s.line_hysteresis_px)
                             if len(snap["danger_line"]) == 2 else None)
        self._zone = (RestrictedZone(snap["zone"])
                      if len(snap["zone"]) >= 3 else None)
        return snap["mode"], snap

    def _sync_analytics(self, snap):
        """Objetos de analítica del HILO DE INFERENCIA (líneas/zonas con
        estado de tracking propio). El worker dibuja instancias aparte:
        la calibración se lee del mismo Calibration pero el estado
        prev_pos de cada track solo vive en el hilo que evalúa cruces."""
        if snap["revision"] == self._an_rev:
            return
        self._an_rev = snap["revision"]
        self._an_count = (CrossingLine(*snap["count_line"], kind="count",
                                      margin_px=self.s.line_hysteresis_px,
                                      invert=self.s.invert_direction)
                          if len(snap["count_line"]) == 2 else None)
        self._an_danger = (CrossingLine(*snap["danger_line"], kind="danger",
                                       margin_px=self.s.line_hysteresis_px)
                           if len(snap["danger_line"]) == 2 else None)
        self._an_zone = (RestrictedZone(snap["zone"])
                         if len(snap["zone"]) >= 3 else None)

    def _infer_loop(self, model):
        """Hilo de inferencia: corre model.track() de forma continua sobre
        el frame más reciente ofrecido por el worker y actualiza las cajas
        compartidas + toda la analítica (conteo, alertas, evidencia).
        Nunca toca el loop de publicación: el visor no espera a YOLO."""
        AW, AH = self.s.analysis_width, self.s.analysis_height
        last_seq = -1
        while not self._infer_stop.is_set():
            while True:
                try:
                    cmd = self._infer_ctl.get_nowait()
                except queue.Empty:
                    break
                if cmd[0] == "set_imgsz":
                    self._imgsz = int(cmd[1])
                elif cmd[0] == "reset_counters":
                    self.total_in = self.total_out = 0
                    self._alert_cooldown.clear()
                    if self._an_count:
                        self._an_count.reset()
                    if self._an_danger:
                        self._an_danger.reset()

            frame, last_seq = self._infer_slot.wait_new(last_seq, timeout=0.5)
            if frame is None:
                continue

            self._sync_analytics(self.calibration.snapshot())

            results = model.track(
                frame,
                persist=True,
                classes=self.s.detect_classes,
                conf=self.s.conf,
                iou=self.s.iou,
                imgsz=self._imgsz,
                tracker=self.s.tracker_cfg,
                verbose=False,
            )
            now = time.time()
            self._det_times.append(now)
            if len(self._det_times) > 2:
                span = self._det_times[-1] - self._det_times[0]
                if span > 0:
                    self._det_fps = (len(self._det_times) - 1) / span

            boxes, track_ids = self._extract(results)
            self.aforo = len(track_ids)

            if len(boxes):
                # Acotar cajas al lienzo de análisis: ni el rectángulo
                # ni el punto-pie pueden sobresalir del frame anotado.
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, AW - 1)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, AH - 1)

            bad = np.zeros(len(track_ids), dtype=bool)
            for i, (box, tid) in enumerate(zip(boxes, track_ids)):
                x1, y1, x2, y2 = map(int, box)
                foot = ((x1 + x2) // 2, y2)

                res_count = (self._an_count.update(tid, foot, AW, AH)
                             if self._an_count else None)
                if res_count == "IN":
                    self.total_in += 1
                    self.writer.enqueue_event("ENTRADA", tid)
                    self._emit("event", msg=f"ID #{tid} ENTRÓ | In: {self.total_in}")
                elif res_count == "OUT":
                    self.total_out += 1
                    self.writer.enqueue_event("SALIDA", tid)
                    self._emit("event", msg=f"ID #{tid} SALIÓ | Out: {self.total_out}")

                danger = (self._an_danger.update(tid, foot, AW, AH) == "CROSS"
                          if self._an_danger else False)
                in_zone = (self._an_zone.contains(foot, AW, AH)
                           if self._an_zone else False)
                if danger or in_zone:
                    bad[i] = True
                    self._last_alert = (tid, now)
                    # 'frame' es copia privada del hilo: evidencia limpia
                    self._trigger_alert(tid, "LINEA_PROHIBIDA" if danger
                                        else "ZONA_PROHIBIDA", frame)

            # Anti-flicker (solo visual): un ID que desaparece de esta
            # inferencia pero existía en la anterior se dibuja 1 ciclo
            # más con su última caja. NO alimenta la analítica (cruces/
            # alertas usan solo detecciones reales de esta pasada) ni el
            # aforo; al siguiente ciclo se descarta si no reaparece.
            cur_ids = set(int(t) for t in track_ids)
            carry_next = {int(t): (b.copy(), bool(bd))
                          for t, b, bd in zip(track_ids, boxes, bad)}
            for tid, (pbox, pbad) in self._carry.items():
                if tid not in cur_ids:
                    boxes = np.vstack([boxes, pbox[None, :]]) if len(boxes) \
                        else pbox[None, :].copy()
                    track_ids = np.append(track_ids, tid)
                    bad = np.append(bad, pbad)
            self._carry = carry_next

            # Publicación atómica de las cajas: el worker lee las refs.
            self._last_boxes, self._last_ids, self._last_bad = \
                boxes, track_ids, bad

    def run(self):
        if self._model is not None:
            model = self._model
        else:
            try:
                model = load_model(
                    self.s,
                    emit=lambda m: self._emit("state", state="loading_model",
                                              msg=m))
            except Exception as e:
                self._emit("state", state="error", msg=f"Modelo: {e}")
                return
        self._emit("state", state="model_ready")

        AW, AH = self.s.analysis_width, self.s.analysis_height
        last_seq = -1
        self._emit("state", state="ok")

        # Hilo de inferencia: corre model.track() de forma continua sobre
        # la copia más reciente que le dejamos en _infer_slot.
        self._infer_stop.clear()
        self._infer_thread = threading.Thread(
            target=self._infer_loop, args=(model,),
            daemon=True, name="InferThread")
        self._infer_thread.start()

        stride = max(1, int(getattr(self.s, "infer_stride", 1)))

        while True:
            if self._drain_controls():
                self._infer_stop.set()
                self._infer_thread.join(timeout=2.0)
                self._emit("state", state="stopped")
                return

            frame, last_seq = self.in_slot.wait_new(last_seq, timeout=0.5)
            if frame is None:
                continue

            # Un solo lienzo de trabajo: se dibuja y publica sobre él.
            # La copia que va al hilo de inferencia es privada (evidencia
            # limpia para alertas + sin race con el dibujo de overlays).
            annotated = cv2.resize(frame, (AW, AH))

            self._infer_tick += 1
            if self._infer_tick >= stride:
                self._infer_tick = 0
                self._infer_slot.set(annotated.copy())

            mode, snap = self._sync_calibration(self.calibration.snapshot())

            boxes, track_ids, bad = \
                self._last_boxes, self._last_ids, self._last_bad

            # --- overlays de calibración ---
            if self._count_line:
                self._count_line.draw(annotated, AW, AH, COLOR_COUNT, "LINEA CONTEO")
            if self._danger_line:
                self._danger_line.draw(annotated, AW, AH, COLOR_DANGER, "CRUCE PROHIBIDO")
            if self._zone:
                self._zone.draw(annotated, AW, AH)
            self._draw_pending_points(annotated, snap, AW, AH)

            # --- detecciones (propias o cacheadas del stride) ---
            for box, tid, is_bad in zip(boxes, track_ids, bad):
                color = COLOR_BAD if is_bad else COLOR_OK
                x1, y1, x2, y2 = map(int, box)
                foot = ((x1 + x2) // 2, y2)
                if is_bad:
                    _tag(annotated, "INTRUSION", (x1, y1 - 34),
                         COLOR_BAD, scale=0.6)
                cv2.rectangle(annotated, (x1, y1), (x2, y2),
                              color, 2, cv2.LINE_AA)
                cv2.circle(annotated, foot, 4, color, -1, cv2.LINE_AA)
                _tag(annotated, f"ID:{tid}", (x1, y1 - 10), color)

            # --- banner de alerta (duración por tiempo, no por frames) ---
            if self._last_alert and time.time() - self._last_alert[1] < self.s.alert_banner_s:
                tid = self._last_alert[0]
                x0, x1 = AW // 2 - 250, AW // 2 + 250
                roi = annotated[12:56, x0:x1]
                chip = np.full(roi.shape, COLOR_BAD, dtype=np.uint8)
                cv2.addWeighted(chip, 0.75, roi, 0.25, 0, roi)
                cv2.rectangle(annotated, (x0, 12), (x1, 55),
                              COLOR_BAD, 2, cv2.LINE_AA)
                cv2.putText(annotated, f"ALERTA: INTRUSION ID #{tid}",
                            (AW // 2 - 230, 45), FONT,
                            0.75, (255, 255, 255), 2, cv2.LINE_AA)

            # --- publicar ---
            # infer_fps = frames publicados/s sobre una ventana real.
            # No usar EMA de 1/período: tras cada inferencia (~70-100 ms)
            # el siguiente frame ya está encolado y sale en ráfaga (~5 ms
            # después) — la media de tasas instantáneas queda sesgada a
            # valores inflados. La ventana cuenta publicaciones/tiempo.
            now = time.time()
            self._pub_times.append(now)
            if len(self._pub_times) > 2:
                span = self._pub_times[-1] - self._pub_times[0]
                rate = (len(self._pub_times) - 1) / span if span > 0 else 0
                self._fps_ema = rate if not self._fps_ema \
                    else 0.85 * self._fps_ema + 0.15 * rate

            # GC periódico (~cada 900 publicaciones ≈ 30 s @30FPS): los
            # buffers temporales (results de ultralytics, arrays vstack)
            # generan basura cíclica; recolectarla de forma acotada evita
            # crecimientos lentos de RSS en sesiones largas sin pausas de
            # GC en el camino caliente de cada frame.
            self._gc_tick += 1
            if self._gc_tick >= 900:
                self._gc_tick = 0
                gc.collect()

            self.out_slot.set({
                "frame": annotated,
                "stats": {
                    "aforo": self.aforo,
                    "in": self.total_in,
                    "out": self.total_out,
                    "infer_fps": self._fps_ema,
                    "det_fps": self._det_fps,
                },
            })

    def _trigger_alert(self, tid, tipo, clean_frame):
        now = time.time()
        last = self._alert_cooldown.get(tid, 0)
        if now - last < self.s.alert_cooldown_s:
            return                                  # cooldown: no spamear
        self._alert_cooldown[tid] = now
        # Poda acotada: sin ella el dict crece con cada track ID nuevo en
        # sesiones largas (fuga de memoria lenta).
        if len(self._alert_cooldown) > 512:
            cutoff = now - self.s.alert_cooldown_s
            self._alert_cooldown = {k: v for k, v in
                                    self._alert_cooldown.items()
                                    if v >= cutoff}
        ruta = new_capture_path(tid, "alerta")
        self.writer.enqueue_alert(tipo, tid, clean_frame, ruta)
        self._emit("event", msg=f"ALERTA {tipo}: ID #{tid} capturado.")
        self._emit("alert", tipo=tipo, tid=tid)

    @staticmethod
    def _extract(results):
        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return np.empty((0, 4)), np.empty((0,), dtype=int)
        return (r.boxes.xyxy.cpu().numpy(),
                r.boxes.id.int().cpu().numpy())

    @staticmethod
    def _draw_pending_points(img, snap, w, h):
        """Puntos del modo de dibujo activo (feedback visual mientras calibrás)."""
        pending = {"linea_conteo": (snap["count_line"], COLOR_COUNT),
                   "linea_alarma": (snap["danger_line"], COLOR_DANGER),
                   "zona": (snap["zone"], COLOR_ZONE)}.get(snap["mode"])
        if not pending:
            return
        pts, color = pending
        for p in pts:
            cv2.circle(img, (int(p[0] * w), int(p[1] * h)), 5,
                       color, -1, cv2.LINE_AA)
