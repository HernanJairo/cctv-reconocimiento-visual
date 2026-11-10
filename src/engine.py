"""Motor de visión: InferWorker.

Consume el frame MÁS RECIENTE del FrameSlot de captura (los intermedios se
descartan → latencia acotada), ejecuta YOLO+ByteTrack sobre un lienzo de
análisis de resolución fija, aplica la analítica (conteo/alarmas/zonas),
dibuja overlays y publica el frame anotado + estadísticas para la GUI.

Nunca toca Tkinter ni hace I/O de disco: la evidencia y los INSERT se
encolan al EventWriter y los mensajes a status_q.
"""
import os
import queue
import threading
import time
from datetime import datetime

import cv2
import numpy as np

# IMPORTANTE: torch y ultralytics se importan aquí (hilo principal, en el
# arranque) y NO dentro del worker. Importar torch o crear el modelo en un
# hilo secundario puede bloquearse en Windows (conflicto de runtimes
# OpenMP/MKL durante la carga perezosa de DLLs y chequeos del SDK).
import torch
from ultralytics import YOLO

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
    """Etiqueta con chip semitransparente: legible sin tapar la imagen."""
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org
    x = max(0, min(x, img.shape[1] - tw - 6))
    y = max(th + 8, min(y, img.shape[0] - 2))
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 4, y - th - 6),
                  (x + tw + 4, y + bl + 2), color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
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
            elif cmd[0] == "set_imgsz":
                self._imgsz = int(cmd[1])

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

        while True:
            if self._drain_controls():
                self._emit("state", state="stopped")
                return

            frame, last_seq = self.in_slot.wait_new(last_seq, timeout=0.5)
            if frame is None:
                continue

            t0 = time.time()
            analysis = cv2.resize(frame, (AW, AH))
            annotated = analysis.copy()          # 'analysis' queda limpio p/evidencia

            mode, snap = self._sync_calibration(self.calibration.snapshot())

            results = model.track(
                analysis,
                persist=True,
                classes=self.s.detect_classes,
                conf=self.s.conf,
                iou=self.s.iou,
                imgsz=self._imgsz,
                tracker=self.s.tracker_cfg,
                verbose=False,
            )

            # --- overlays de calibración ---
            if self._count_line:
                self._count_line.draw(annotated, AW, AH, COLOR_COUNT, "LINEA CONTEO")
            if self._danger_line:
                self._danger_line.draw(annotated, AW, AH, COLOR_DANGER, "CRUCE PROHIBIDO")
            if self._zone:
                self._zone.draw(annotated, AW, AH)
            self._draw_pending_points(annotated, snap, AW, AH)

            # --- detecciones ---
            boxes, track_ids = self._extract(results)
            self.aforo = len(track_ids)

            if len(boxes):
                # Acotar cajas al lienzo de análisis: ni el rectángulo ni
                # el punto-pie pueden sobresalir del frame anotado.
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, AW - 1)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, AH - 1)

            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                foot = ((x1 + x2) // 2, y2)

                res_count = (self._count_line.update(tid, foot, AW, AH)
                             if self._count_line else None)
                if res_count == "IN":
                    self.total_in += 1
                    self.writer.enqueue_event("ENTRADA", tid)
                    self._emit("event", msg=f"ID #{tid} ENTRÓ | In: {self.total_in}")
                elif res_count == "OUT":
                    self.total_out += 1
                    self.writer.enqueue_event("SALIDA", tid)
                    self._emit("event", msg=f"ID #{tid} SALIÓ | Out: {self.total_out}")

                danger = (self._danger_line.update(tid, foot, AW, AH) == "CROSS"
                          if self._danger_line else False)
                in_zone = (self._zone.contains(foot, AW, AH)
                           if self._zone else False)

                color = COLOR_OK
                if danger or in_zone:
                    color = COLOR_BAD
                    self._last_alert = (tid, time.time())
                    self._trigger_alert(tid, "LINEA_PROHIBIDA" if danger
                                        else "ZONA_PROHIBIDA", analysis)
                    _tag(annotated, "INTRUSION", (x1, y1 - 34),
                         COLOR_BAD, scale=0.6)

                cv2.rectangle(annotated, (x1, y1), (x2, y2),
                              color, 2, cv2.LINE_AA)
                cv2.circle(annotated, foot, 4, color, -1, cv2.LINE_AA)
                _tag(annotated, f"ID:{tid}", (x1, y1 - 10), color)

            # --- banner de alerta (duración por tiempo, no por frames) ---
            if self._last_alert and time.time() - self._last_alert[1] < self.s.alert_banner_s:
                tid = self._last_alert[0]
                overlay = annotated.copy()
                cv2.rectangle(overlay, (AW // 2 - 250, 12), (AW // 2 + 250, 55),
                              COLOR_BAD, -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
                cv2.rectangle(annotated, (AW // 2 - 250, 12), (AW // 2 + 250, 55),
                              COLOR_BAD, 2, cv2.LINE_AA)
                cv2.putText(annotated, f"ALERTA: INTRUSION ID #{tid}",
                            (AW // 2 - 230, 45), FONT,
                            0.75, (255, 255, 255), 2, cv2.LINE_AA)

            # --- publicar ---
            dt = time.time() - t0
            inst_fps = 1.0 / dt if dt > 0 else 0
            self._fps_ema = inst_fps if not self._fps_ema \
                else 0.9 * self._fps_ema + 0.1 * inst_fps

            self.out_slot.set({
                "frame": annotated,
                "stats": {
                    "aforo": self.aforo,
                    "in": self.total_in,
                    "out": self.total_out,
                    "infer_fps": self._fps_ema,
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
