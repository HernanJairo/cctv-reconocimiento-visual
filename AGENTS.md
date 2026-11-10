# CCTV Inteligente — Videovigilancia y Auditoría Forense

## Ejecutar
```
python main.py                  # lanzar la aplicación (único entrypoint)
python main.py --export-onnx    # regenerar models/yolov8n.onnx
```

## Estructura
- `main.py` — entrypoint; carga settings + modelo en hilo principal
- `config/settings.yaml` — toda la configuración (modelo, imgsz, fuentes, FFmpeg flags)
- `src/paths.py` — rutas absolutas (BASE_DIR); nunca usar rutas relativas
- `src/config.py` — dataclass Settings + loader YAML
- `src/capture.py` — CaptureThread + FrameSlot (latest-frame-wins, watchdog, reconexión, DVR)
- `src/engine.py` — InferWorker (YOLO+ByteTrack, analítica, overlays) + load_model()
- `src/geometry.py` — CrossingLine (histéresis), RestrictedZone, Calibration (thread-safe)
- `src/storage.py` — EventWriter (fotos+SQLite async, WAL), galería, CSV
- `src/gui/app.py` — SecurityApp (2 tabs: live + galería forense)
- `src/gui/viewer.py` — ImageViewerModal (zoom, flechas, Esc)
- `models/` — yolov8n.pt, yolov8n.onnx, yolov8s.pt
- `data/` — seguridad.db, capturas/, videos/, reportes/

## Reglas de arquitectura (NO romper)
1. **Tkinter solo en el hilo del mainloop.** Los workers comunican por
   FrameSlot/queues; la GUI drena con `after()`. Nunca llamar
   `widget.configure` desde un thread.
2. **torch/ultralytics se importan y el modelo se carga + warm-up en el
   hilo principal** (main.py → `load_model()`), ANTES de abrir cualquier
   `cv2.VideoCapture`. Importar torch o hacer la 1ra inferencia en un
   thread con FFmpeg decodificando se BLOQUEA en Windows (conflicto
   OpenMP/MKL). Por eso `src/engine.py` importa torch a nivel módulo.
3. Tracker fijo: `bytetrack.yaml`. NO usar botsort (warnings de GMC en
   consola y más lento en CPU).
4. I/O de disco (cv2.imwrite, SQLite) solo vía EventWriter — nunca inline
   en el loop de inferencia.
5. Calibración siempre en coordenadas normalizadas [0..1] sobre el lienzo
   de análisis (analysis_width × analysis_height), nunca en píxeles de
   widget ni de la fuente original.
