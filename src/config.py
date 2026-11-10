"""Configuración del sistema.

Carga config/settings.yaml sobre valores por defecto seguros. Si el
archivo falta o PyYAML no está disponible, la app arranca igual.
"""
import os
from dataclasses import dataclass, field, fields

from .paths import SETTINGS_PATH, MODELS_DIR, VIDEOS_DIR, CONFIG_DIR


@dataclass
class Settings:
    # --- Modelo / inferencia ---
    model_file: str = "yolov8n.pt"        # puede ser yolov8n.onnx
    imgsz: int = 512
    imgsz_stream: int = 512               # streams IP: menos carga por inferencia
    conf: float = 0.35                    # estricto: corta falsos positivos
    iou: float = 0.55                     # NMS: separa personas sin duplicar
    tracker: str = "bytetrack.yaml"       # evita warnings GMC de BoT-SORT
    torch_threads: int = 4                # deja hilos libres para captura+GUI
    detect_classes: list = field(default_factory=lambda: [0])  # personas

    # --- Pipeline ---
    analysis_width: int = 960             # lienzo interno fijo de analítica
    analysis_height: int = 540
    render_fps: int = 30                  # refresco de GUI (after)
    jpeg_quality: int = 90

    # --- Analítica ---
    line_hysteresis_px: int = 5           # banda muerta anti-doble-conteo
    alert_cooldown_s: float = 8.0         # mínimo entre alertas del mismo ID
    alert_banner_s: float = 2.5
    invert_direction: bool = False        # invertir ENTRADA/SALIDA

    # --- Fuentes ---
    webcam_index: int = 0
    webcam_width: int = 640
    webcam_height: int = 480
    webcam_fps: int = 30
    webcam_mirror: bool = True            # flip horizontal solo en webcam
    default_video: str = field(
        default_factory=lambda: os.path.join(VIDEOS_DIR, "prueba.mp4"))
    default_stream_url: str = "http://192.168.100.XX:8080/video"

    # Opciones FFmpeg para streams IP/RTSP (baja latencia + reconexión).
    # Formato OpenCV: pares "clave;valor" separados por "|".
    ffmpeg_options: str = (
        "rtsp_transport;tcp"
        "|stimeout;5000000"
        "|max_delay;500000"
        "|fflags;nobuffer"
        "|flags;low_delay"
        "|probesize;32"
        "|analyzeduration;0"
        "|reconnect;1"
        "|reconnect_streamed;1"
        "|reconnect_delay_max;2"
    )

    # --- Captura de baja latencia (streams IP / webcam) ---
    # Vacía el backlog del decodificador: latest-frame-wins también
    # dentro de FFmpeg/driver, no solo en el FrameSlot.
    low_latency: bool = True
    drain_max_grabs: int = 30             # tope de frames viejos por lectura
    fresh_grab_ms: float = 25.0           # grab() >= esto = esperó a la red

    read_fail_threshold: int = 30         # lecturas fallidas antes de reconectar
    reconnect_backoff_s: tuple = (1.0, 2.0, 4.0, 8.0, 10.0)

    @property
    def model_path(self) -> str:
        return self.model_file if os.path.isabs(self.model_file) \
            else os.path.join(MODELS_DIR, self.model_file)

    @property
    def tracker_cfg(self) -> str:
        """YAML del tracker: nombre interno de ultralytics o archivo en config/."""
        if os.path.isabs(self.tracker):
            return self.tracker
        custom = os.path.join(CONFIG_DIR, self.tracker)
        return custom if os.path.exists(custom) else self.tracker


def load_settings() -> Settings:
    s = Settings()
    try:
        import yaml
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(Settings)}
        for k, v in data.items():
            if k in valid and v is not None:
                if k == "reconnect_backoff_s":
                    v = tuple(v)
                setattr(s, k, v)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[config] settings.yaml inválido, usando defaults: {e}")
    return s
