"""Banco de videos locales de demostración — cero dependencia de red.

Los clips de settings.video_presets viven versionados en data/videos/.
`ensure_clip()` / `ensure_demo_clips()` garantizan que existan:

1. Si el archivo ya existe, se devuelve tal cual (caso normal: viene en
   el repo/ZIP).
2. Si falta, se sintetiza localmente un clip estilo CCTV con OpenCV
   (figuras en movimiento + timestamp) — la fuente abre siempre, aunque
   el detector no marque personas sobre el clip sintético. NUNCA se
   intenta descargar nada: el sistema arranca 100% offline.
"""
import os
from datetime import datetime

import cv2
import numpy as np

from .paths import VIDEOS_DIR

W, H, FPS, SECONDS = 640, 360, 20, 15


def _person(img, cx, base_y, scale=1.0):
    """Silueta humana simple (cabeza + torso + piernas) para el clip
    sintético de respaldo."""
    s = scale
    cv2.ellipse(img, (cx, base_y - int(95 * s)), (int(11 * s), int(13 * s)),
                0, 0, 360, (200, 210, 220), -1, cv2.LINE_AA)
    cv2.rectangle(img, (cx - int(14 * s), base_y - int(80 * s)),
                  (cx + int(14 * s), base_y - int(40 * s)),
                  (180, 190, 205), -1, cv2.LINE_AA)
    cv2.rectangle(img, (cx - int(13 * s), base_y - int(40 * s)),
                  (cx - int(2 * s), base_y), (160, 165, 180), -1, cv2.LINE_AA)
    cv2.rectangle(img, (cx + int(3 * s), base_y - int(40 * s)),
                  (cx + int(14 * s), base_y), (160, 165, 180), -1, cv2.LINE_AA)


def _synthesize_clip(dest, variant=0):
    """Genera un clip estilo CCTV: pasillo oscuro, dos figuras cruzando
    en sentidos opuestos en bucle, ruido y timestamp."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    vw = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rng = np.random.default_rng(7 + variant)
    n = FPS * SECONDS
    for i in range(n):
        t = ((i / n) + variant * 0.37) % 1.0     # fase distinta por clip
        img = np.full((H, W, 3), (28, 32, 38), np.uint8)
        cv2.rectangle(img, (0, int(H * 0.55)), (W, H), (38, 42, 48), -1)
        for x in range(0, W, 40):                # juntas del "piso"
            cv2.line(img, (x, int(H * 0.55)), (x + 18, H), (50, 54, 60), 1)
        img = cv2.add(img, rng.integers(0, 14, img.shape, dtype=np.uint8))
        _person(img, int(W * (0.08 + 0.84 * t)), int(H * 0.88))          # →
        _person(img, int(W * (0.92 - 0.84 * t)), int(H * 0.70), 0.75)    # ←
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, f"CAM-SIM  {ts}", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 120), 1,
                    cv2.LINE_AA)
        cv2.putText(img, "SIMULACION", (W - 130, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 200), 1,
                    cv2.LINE_AA)
        vw.write(img)
    vw.release()


def _resolve(preset_or_path, variant=0):
    """Devuelve la ruta absoluta del clip; lo sintetiza si falta."""
    path = preset_or_path if isinstance(preset_or_path, str) \
        else preset_or_path.get("file", "")
    if not os.path.isabs(path):
        path = os.path.join(VIDEOS_DIR, path)
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        _synthesize_clip(path, variant=variant)
    return path


def ensure_clip(preset):
    """Garantiza el clip local de un preset {'file': ...}.
    Devuelve ruta absoluta; sintetiza solo si el archivo falta."""
    return _resolve(preset)


def ensure_demo_clips(presets):
    """Asegura todos los clips de la lista de presets (settings.video_presets)."""
    for i, p in enumerate(presets):
        _resolve(p, variant=i)


def ensure_demo_video(path=None):
    """Devuelve la ruta de un video local usable como fuente.
    Lo sintetiza solo si no existe o está vacío."""
    return _resolve(path or "prueba.mp4")
