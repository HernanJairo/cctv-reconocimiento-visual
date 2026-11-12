"""Sistema Integral de Videovigilancia y Auditoría Forense.

Punto de entrada único:
    python main.py                  → lanzar la aplicación
    python main.py --export-onnx    → exportar el .pt configurado a ONNX
"""
import argparse
import os
import sys

# Las opciones FFmpeg deben estar en el entorno ANTES de que OpenCV
# inicialice su backend de captura.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|timeout;8000000|rw_timeout;8000000"
    "|max_delay;500000|fflags;nobuffer|flags;low_delay|probesize;32"
    "|analyzeduration;0"
    "|reconnect;1|reconnect_streamed;1|reconnect_delay_max;2")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import load_settings  # noqa: E402
from src.paths import MODELS_DIR  # noqa: E402


def export_onnx():
    from ultralytics import YOLO
    s = load_settings()
    src = s.model_path
    dst = os.path.splitext(src)[0] + ".onnx"
    print(f"Exportando {src} → ONNX (imgsz={s.imgsz})…")
    YOLO(src).export(format="onnx", imgsz=s.imgsz, simplify=True, opset=12)
    print(f"Listo: {dst}. Para usarlo, en config/settings.yaml ponga: "
          f"model_file: {os.path.basename(dst)}")


def main():
    ap = argparse.ArgumentParser(description="CCTV Inteligente — Videovigilancia")
    ap.add_argument("--export-onnx", action="store_true",
                    help="exporta el .pt configurado a ONNX y sale")
    args = ap.parse_args()

    if args.export_onnx:
        export_onnx()
        return

    settings = load_settings()

    # Desactiva el sync/analytics online de Ultralytics (evita esperas de
    # red durante la carga del modelo en equipos sin internet).
    try:
        os.environ.setdefault("ULTRALYTICS_ANALYTICS", "false")
        from ultralytics import settings as _yolo_settings
        _yolo_settings.update({"sync": False})
    except Exception:
        pass

    # El modelo se carga y calienta en el hilo principal ANTES de abrir la
    # GUI o cualquier captura de video (evita bloqueos OpenMP/FFmpeg).
    from src.engine import load_model
    model = load_model(settings)

    import customtkinter as ctk
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")

    from src.gui.app import SecurityApp
    app = SecurityApp(settings, model=model)
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
