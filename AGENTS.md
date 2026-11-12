# CCTV Inteligente — Videovigilancia y Auditoría Forense

## Ejecutar
```
iniciar.bat                          # lanzador 1-clic (crea venv, instala deps, descarga modelo)
python main.py                       # lanzar la aplicación (único entrypoint)
python main.py --export-onnx         # regenerar models/yolov8n.onnx
python tests/benchmark_pipeline.py   # benchmark 60 s: fps, latencia e2e, RAM
python tests/test_line_crossing.py   # test cruce de línea anti-salto (sin modelo ni GUI)
python test_fps_e2e.py               # FPS real del visor con pipeline completo (~35 s)
python tests/audit_streams.py        # audita URLs de stream con cv2 (~2 min, paralelo)
```

## Fuentes de video (100% local / sin internet)
- `data/videos/` — banco de clips versionados: prueba.mp4 (personas
  interior), peatones_calle.mp4 (vista elevada), retail.mp4 (tienda),
  zona_trabajo.mp4 (zona elevada). Todos ≥30 FPS fuente (los de 10-12
  FPS se re-encodaron a 30 con frames duplicados: misma duración real,
  visor ≥24 FPS). Se reproducen en bucle infinito.
- `settings.video_presets` — presets del menú "🎬 Videos de prueba";
  `src/demo_video.py` los garantiza (síntesis local si faltan — nunca
  descarga de red).
- La fuente "Cámara de Celular / Stream IP" usa URL local (RTSP/HTTP
  LAN); NO hay presets de internet público.

## Estructura
- `main.py` — entrypoint; carga settings + modelo en hilo principal
- `config/settings.yaml` — toda la configuración (modelo, imgsz, fuentes, FFmpeg flags)
- `src/paths.py` — rutas absolutas (BASE_DIR); nunca usar rutas relativas
- `src/config.py` — dataclass Settings + loader YAML
- `src/capture.py` — CaptureThread + FrameSlot (latest-frame-wins, watchdog, reconexión, DVR)
- `src/engine.py` — InferWorker (YOLO+ByteTrack, analítica, overlays) + load_model()
- `src/geometry.py` — CrossingLine (histéresis), RestrictedZone, Calibration (thread-safe)
- `src/storage.py` — EventWriter (fotos+SQLite async, WAL), galería, CSV
- `src/demo_video.py` — ensure_clip/ensure_demo_clips: banco de clips locales infalible
- `src/gui/app.py` — SecurityApp (2 tabs: live + galería forense)
- `src/gui/viewer.py` — ImageViewerModal (zoom, flechas, Esc)
- `models/` — yolov8s.pt (único modelo versionado)
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
6. **Inferencia ASÍNCRONA** (`engine.py`): `InferWorker` publica a la
   cadencia de la fuente (visor 30-60 FPS) y ofrece 1 de cada
   `infer_stride` frames a `_infer_slot`; `InferThread` corre
   `model.track` continuo sobre el más reciente y actualiza
   `_last_boxes/_last_ids/_last_bad` + toda la analítica (líneas/zonas
   propias del hilo: `_an_*`). NO volver a inferencia síncrona en el loop
   de publicación: bloquea el visor a ~T_inf/stride FPS.
7. El cruce de línea se evalúa solo en frames de inferencia — la
   intersección segmento-segmento de CrossingLine no pierde cruces.
