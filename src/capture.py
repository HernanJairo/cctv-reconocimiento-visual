"""Captura de video desacoplada.

FrameSlot      : buzón "latest-frame-wins" (el consumidor siempre recibe
                 el frame más nuevo; los intermedios se descartan).
CaptureThread  : hilo dueño del cv2.VideoCapture. Maneja:
                   - apertura por tipo de fuente (DSHOW / FFmpeg / archivo)
                   - watchdog de lecturas fallidas + reconexión con backoff
                   - DVR de archivos: pacing, seek absoluto/relativo, pausa,
                     velocidad — todo por una cola de control (sin races).
                   - reporte de estado/progreso por status_q hacia la GUI.
"""
import os
import queue
import threading
import time

import cv2


class FrameSlot:
    """Buzón de un solo slot con secuencia: el último frame pisa al anterior."""

    def __init__(self):
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0

    def set(self, frame):
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._cond.notify_all()

    def latest(self):
        with self._cond:
            return self._frame, self._seq

    def wait_new(self, last_seq, timeout=1.0):
        """Espera un frame con seq > last_seq. Devuelve (frame, seq) o (None, last_seq)."""
        with self._cond:
            if self._seq <= last_seq:
                self._cond.wait(timeout=timeout)
            if self._seq > last_seq:
                return self._frame, self._seq
            return None, last_seq


class CaptureThread(threading.Thread):
    """
    spec: {"kind": "webcam", "index": 0}
        | {"kind": "stream", "url": "http://.../rtsp://..."}
        | {"kind": "file",   "path": "video.mp4"}

    Comandos de control (cola `ctl`): tuplas
      ("pause", bool) ("seek_rel", segundos) ("seek_abs", frame)
      ("speed", float) ("stop",)
    """

    def __init__(self, spec, settings, frame_slot, status_q):
        super().__init__(daemon=True, name="CaptureThread")
        self.spec = spec
        self.s = settings
        self.slot = frame_slot
        self.status_q = status_q
        self.ctl = queue.Queue()
        self.cap = None
        self.is_file = spec["kind"] == "file"
        # Modo espejo: solo aplica a la webcam integrada (no archivos ni IP).
        self._mirror = spec["kind"] == "webcam" and self.s.webcam_mirror
        self.total_frames = 0
        self.src_fps = 30.0
        self._paused = False
        self._speed = 1.0

    # --- API (GUI) ---
    def send(self, *cmd):
        self.ctl.put(cmd)

    def stop(self):
        self.ctl.put(("stop",))

    # --- helpers internos ---
    def _emit(self, kind, **kw):
        try:
            self.status_q.put_nowait({"src": "capture", "kind": kind, **kw})
        except queue.Full:
            pass

    def _open(self):
        kind = self.spec["kind"]
        if kind == "webcam":
            idx = int(self.spec.get("index", 0))
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.s.webcam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.s.webcam_height)
            cap.set(cv2.CAP_PROP_FPS, self.s.webcam_fps)
        elif kind == "stream":
            # Flags FFmpeg de baja latencia y reconexión: deben estar en el
            # entorno ANTES de crear el VideoCapture.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = self.s.ffmpeg_options
            cap = cv2.VideoCapture(self.spec["url"], cv2.CAP_FFMPEG)
        else:  # file
            cap = cv2.VideoCapture(self.spec["path"])
        return cap

    def _drain_controls(self):
        stop = False
        seeked = False
        while True:
            try:
                cmd = self.ctl.get_nowait()
            except queue.Empty:
                break
            name = cmd[0]
            if name == "stop":
                stop = True
            elif name == "pause":
                self._paused = bool(cmd[1])
                self._emit("state", state="paused" if self._paused else "ok")
            elif name == "speed":
                self._speed = float(cmd[1])
            elif name == "seek_rel" and self.is_file and self.cap:
                cur = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                tgt = max(0, min(self.total_frames - 1,
                                 cur + cmd[1] * self.src_fps))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, tgt)
                seeked = True
            elif name == "seek_abs" and self.is_file and self.cap:
                tgt = max(0, min(self.total_frames - 1, cmd[1]))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, tgt)
                seeked = True
        if seeked and self.cap and self.is_file:
            self._after_seek()
        return stop

    def _after_seek(self):
        """Post-salto: cap.set() ya purga el buffer del decodificador; aquí
        empujamos el frame destino al slot (clave en pausa, donde el loop
        no lee) y reportamos la posición real para que la GUI y el slider
        queden sincronizados en el segundo exacto elegido."""
        if self._paused:
            ret, frame = self.cap.read()
            if ret:
                if self._mirror:
                    frame = cv2.flip(frame, 1)
                self.slot.set(frame)
        cur = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self._emit("progress", frame=cur,
                   total=self.total_frames, fps=self.src_fps)

    def run(self):
        backoff_iter = iter(self.s.reconnect_backoff_s)
        fails = 0
        while True:
            # --- apertura / reapertura ---
            self._emit("state", state="connecting")
            try:
                self.cap = self._open()
            except Exception:
                self.cap = None

            if self.cap is None or not self.cap.isOpened():
                if self.is_file or self.spec["kind"] == "webcam":
                    self._emit("state", state="error",
                               msg="No se pudo abrir la fuente.")
                    return
                # stream: reintentar con backoff
                wait = next(backoff_iter, self.s.reconnect_backoff_s[-1])
                self._emit("state", state="reconnecting",
                           msg=f"Reintentando en {wait:.0f}s…")
                if self._sleep_or_stop(wait):
                    return
                continue

            backoff_iter = iter(self.s.reconnect_backoff_s)
            fails = 0
            if self.is_file:
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._emit("state", state="ok",
                       total_frames=self.total_frames, fps=self.src_fps,
                       is_file=self.is_file)

            # --- loop de lectura ---
            t_prev = time.time()
            while True:
                if self._drain_controls():
                    self._release()
                    self._emit("state", state="stopped")
                    return
                if self._paused:
                    time.sleep(0.04)
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    fails += 1
                    if self.is_file:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        fails = 0
                        continue
                    if fails >= self.s.read_fail_threshold:
                        self._emit("state", state="reconnecting",
                                   msg="Señal perdida. Reconectando…")
                        break          # sale al ciclo de reapertura
                    time.sleep(0.02)
                    continue

                fails = 0
                if self._mirror:
                    frame = cv2.flip(frame, 1)
                self.slot.set(frame)

                if self.is_file:
                    cur = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    self._emit("progress", frame=cur,
                               total=self.total_frames, fps=self.src_fps)
                    # pacing: mantener velocidad de reproducción del archivo
                    period = 1.0 / (self.src_fps * self._speed)
                    dt = time.time() - t_prev
                    if dt < period:
                        time.sleep(period - dt)
                t_prev = time.time()

            self._release()
            wait = next(backoff_iter, self.s.reconnect_backoff_s[-1])
            if self._sleep_or_stop(wait):
                return

    def _sleep_or_stop(self, seconds):
        t0 = time.time()
        while time.time() - t0 < seconds:
            if self._drain_controls():
                return True
            time.sleep(0.05)
        return False

    def _release(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.cap = None
