"""Smoke test: config de detección, construcción de la GUI y seek DVR."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 1) Config de detección
from src.config import load_settings
s = load_settings()
assert s.iou == 0.50, s.iou
assert s.conf == 0.20, s.conf
assert s.imgsz == 640, s.imgsz
assert s.imgsz_stream in (480, 512, 640), s.imgsz_stream
assert s.infer_stride == 3, s.infer_stride
assert s.model_file == "yolov8s.pt", s.model_file
assert s.detect_classes == [0], s.detect_classes
assert s.low_latency and s.drain_max_grabs == 30
assert s.tracker_cfg.endswith("bytetrack_cctv.yaml"), s.tracker_cfg
assert s.video_presets and all(
    p.get("name") and p.get("file") for p in s.video_presets)
# torch_threads autodetectado: entre 1 y 4 según la CPU del equipo
assert 1 <= s.torch_threads <= 4, s.torch_threads
import yaml
with open(s.tracker_cfg, encoding="utf-8") as f:
    trk = yaml.safe_load(f)
assert trk["match_thresh"] == 0.65, trk
assert trk["track_buffer"] == 60, trk
assert trk["new_track_thresh"] == 0.20, trk
assert trk["track_low_thresh"] == 0.10, trk
print("1) Config detección/tracker OK:", s.iou, s.conf, s.imgsz,
      s.model_file, "| match", trk["match_thresh"],
      "| new_track", trk["new_track_thresh"])

# 2) Construcción de la GUI (sin modelo: el worker lo carga en bg)
import customtkinter as ctk
ctk.set_appearance_mode("Dark")
from src.gui.app import SecurityApp, SOURCES
app = SecurityApp(s, model=None)
app.update()
for w in ("lbl_live", "lbl_clock", "combo_source", "entry_src",
          "btn_browse", "menu_presets", "timeline",
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

# 6) Banco de videos locales: cada preset resuelve a un archivo real
#    existente y arma spec de archivo + fuente "Archivo de Video"
assert app.menu_presets.get() == "🎬 Videos de prueba"
for p in s.video_presets:
    app._load_video_preset(p["name"])
    assert app.combo_source.get() == SOURCES[2], app.combo_source.get()
    local_path = app.entry_src.get()
    assert os.path.isabs(local_path) and os.path.exists(local_path), \
        f"{p['name']} -> {local_path}"
    assert app._build_spec() == {"kind": "file", "path": local_path}
print("6) Todos los videos de prueba resuelven a archivo local:",
      len(s.video_presets), "clips")

# 7) Selección de archivo (simulada): fuente archivo + spec correcto
app.combo_source.set(SOURCES[2])
app.entry_src.delete(0, "end")
app.entry_src.insert(0, "C:/tmp/video.mp4")
assert app._build_spec() == {"kind": "file", "path": "C:/tmp/video.mp4"}
print("7) Fuente 'Archivo de Video' arma el spec correcto")

app.on_closing()
print("TODOS LOS TESTS PASARON")
