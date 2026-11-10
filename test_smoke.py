"""Smoke test: config de detección, construcción de la GUI y seek DVR."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 1) Config de detección
from src.config import load_settings
s = load_settings()
assert s.iou == 0.65, s.iou
assert s.conf == 0.15, s.conf
assert s.imgsz == 576, s.imgsz
assert s.model_file == "yolov8s.pt", s.model_file
assert s.tracker_cfg.endswith("bytetrack_cctv.yaml"), s.tracker_cfg
import yaml
with open(s.tracker_cfg, encoding="utf-8") as f:
    trk = yaml.safe_load(f)
assert trk["match_thresh"] == 0.60, trk
assert trk["track_buffer"] == 60, trk
print("1) Config detección/tracker OK:", s.iou, s.conf, s.imgsz,
      s.model_file, "| match", trk["match_thresh"])

# 2) Construcción de la GUI (sin modelo: el worker lo carga en bg)
import customtkinter as ctk
ctk.set_appearance_mode("Dark")
from src.gui.app import SecurityApp
app = SecurityApp(s, model=None)
app.update()
for w in ("lbl_live", "lbl_clock", "combo_source", "timeline",
          "seg_speed", "btn_pause", "lbl_aforo", "lbl_in", "lbl_out",
          "lbl_alertas", "log_box"):
    assert hasattr(app, w), f"falta widget {w}"
print("2) GUI construida con todos los widgets nuevos")

# 3) Seek DVR: press congela progreso, release envía UN seek_abs
app._is_file = True
app._dvr_total, app._dvr_fps = 3000, 30.0


class FakeCap:
    total_frames = 3000
    def __init__(self):
        self.cmds = []
    def send(self, *c):
        self.cmds.append(c)
    def stop(self):
        pass
    def join(self, timeout=None):
        pass


app.running = True
app.cap_thread = FakeCap()

# simular progreso mientras NO hay drag → el slider se mueve
app._is_seeking = False
app._handle_status({"src": "capture", "kind": "progress",
                    "frame": 1500, "total": 3000, "fps": 30})
v_play = app.timeline.get()

# press → drag: el progreso NO debe mover el slider
app._on_seek_press(None)
app.timeline.set(80.0)          # posición del thumb bajo el mouse
app._handle_status({"src": "capture", "kind": "progress",
                    "frame": 2000, "total": 3000, "fps": 30})
assert app.timeline.get() == 80.0, "el progreso movió el slider durante el drag"
print("3) Durante el drag el slider NO sigue al playback (OK)")

# release → un único seek_abs al frame destino
app._on_seek_release(None)
assert app.cap_thread.cmds == [("seek_abs", 2400)], app.cap_thread.cmds
assert not app._is_seeking
print("4) Al soltar: seek_abs(2400) enviado una sola vez (OK)")

# pulso LIVE + reloj corriendo
app._tick_topbar()
assert app.lbl_live.cget("text") == "● LIVE"
app.running = False
app._tick_topbar()
assert app.lbl_live.cget("text") == "● OFFLINE"
print("5) Indicador LIVE pulsante + reloj OK")

app.on_closing()
print("TODOS LOS TESTS PASARON")
