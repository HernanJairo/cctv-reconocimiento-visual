"""Test e2e: pipeline real (captura + YOLO + render) sobre prueba.mp4.
Mide el FPS que el visor reporta en pantalla (lbl_fps)."""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import customtkinter as ctk
ctk.set_appearance_mode("Dark")

from src.config import load_settings
from src.engine import load_model          # main thread (regla OpenMP/FFmpeg)

s = load_settings()
model = load_model(s)

from src.gui.app import SecurityApp
app = SecurityApp(s, model=model)
app.update()

# Fuente: clip cenital de 30 FPS del banco local — caso crítico de
# detección a distancia. El FPS del visor nunca supera la cadencia
# del propio video fuente.
preset60 = next(p for p in s.video_presets if p["file"] == "peatones_calle.mp4")
app._load_video_preset(preset60["name"])
app._toggle_stream()
assert app.running, "el pipeline no arrancó"

fps_seen = []
seq0 = None
t0 = time.time()
while time.time() - t0 < 20:
    app.update()
    m = re.search(r"([\d.]+)", app.lbl_fps.cget("text"))
    if m:
        fps_seen.append(float(m.group(1)))
    if seq0 is None and app._last_out_seq > 0:
        seq0 = (app._last_out_seq, time.time())
    time.sleep(0.04)

if app.cap_thread is not None:
    print(f"src_fps detectado: {app.cap_thread.src_fps}")
if seq0 is not None:
    real_fps = (app._last_out_seq - seq0[0]) / (time.time() - seq0[1])
    print(f"FPS real entregado (Δseq/s consumido por GUI): {real_fps:.1f}")

tail = fps_seen[len(fps_seen) // 2:]          # 2da mitad: régimen estable
avg = sum(tail) / len(tail) if tail else 0.0
print(f"FPS visor: min={min(tail):.1f} avg={avg:.1f} "
      f"max={max(tail):.1f} ({len(tail)} muestras)")

app._stop_pipeline()
app.on_closing()

assert tail, "el visor nunca reportó FPS"
assert avg >= 24.0, f"FPS insuficiente: {avg:.1f} < 24"
print(f"PASS: visor reporta {avg:.1f} FPS >= 24 en régimen estable")
