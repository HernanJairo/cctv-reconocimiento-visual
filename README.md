@'
# 📹 Sistema Integral de Videovigilancia y Auditoría Forense CCTV

Sistema de visión artificial y analítica perimetral en tiempo real desarrollado en Python. Implementa detección, seguimiento continuo (*tracking*), aforo, detección de intrusión en líneas/zonas y registro forense asíncrono con base de datos SQLite y visor interactivo.

El proyecto fue diseñado y optimizado específicamente para operar de forma fluida (30 FPS sostenidos) en entornos sin GPU dedicada (procesadores CPU AMD Ryzen / Intel de notebooks).

---

## 📌 Bitácora de Desarrollo: De prototipo inestable a arquitectura profesional

### 1. El punto de partida y los primeros bloqueos
El proyecto inició como un prototipo (`tp_vision`) con una arquitectura monolítica y síncrona:
* **El problema de fluidez:** Al intentar procesar cada cuadro en una notebook (AMD Ryzen 5 PRO 5675U con gráficos integrados), la inferencia directa de YOLOv8 junto con el redibujado de la interfaz y la escritura en disco congelaba la aplicación, cayendo a 5–10 FPS con tirones severos.
* **Advertencias masivas en consola:** Aparecían constantes avisos `WARNING not enough matching points` y fallos de `GMC` (Global Motion Compensation).
* **Falsos conteos y descalibración:** Al marcar líneas en pantalla, los clics quedaban desfasados hacia los costados por problemas de escalado de pantalla (DPI), y el cruce contaba dobles o perdía personas si caminaban juntas o de espaldas.

### 2. El proceso de optimización asistida por IA (Gemini + Devin)
Para resolver los límites del hardware sin perder precisión, se implementó un flujo de trabajo iterativo utilizando herramientas de Inteligencia Artificial:
* **Gemini (Arquitectura, Diagnóstico y Diseño de Estrategias):**
  * Identificó el motivo de las advertencias: el tracker por defecto era BoT-SORT (diseñado para cámaras en movimiento), requiriendo el cambio a **ByteTrack** para cámaras fijas.
  * Diseñó el desacople por hilos (*multithreading*) con política *latest-frame-wins*, aislando la captura, la inferencia de IA y el guardado en disco de la interfaz de usuario.
  * Formuló la lógica geométrica por producto cruz con histéresis y la compensación de coordenadas para eliminar el desvío del ratón.
* **Devin (Ejecución, Refactorización en Código y Debugging a Bajo Nivel):**
  * Reorganizó la estructura de archivos en un esquema limpio (`src/`, `models/`, `data/`, `config/`, `main.py`).
  * Realizó pruebas de rendimiento comparando PyTorch vs. ONNX Runtime directamente sobre el procesador.
  * **Resolución de un bug crítico en Windows:** Detectó que inicializar PyTorch en un hilo secundario mientras OpenCV/FFmpeg abría el video producía un bloqueo de carga de DLLs (conflicto de runtimes OpenMP/MKL). Lo solucionó implementando un *warm-up* determinístico en el hilo principal antes de activar la captura.

---

## 🛠️ Stack Tecnológico y Justificación

* **Python 3.14:** Lenguaje base para prototipado rápido e integración de librerías científicas.
* **Ultralytics YOLOv8 (Modelo Small `yolov8s.pt` / Nano):** Balance óptimo entre capacidad de generalización y consumo de recursos. Con `yolov8s` y resolución calibrada a 576/640 px se garantiza la detección de personas pequeñas o lejanas sin penalizar la CPU.
* **ByteTrack:** Algoritmo de seguimiento basado en filtros de Kalman y coincidencia en dos etapas. Mantiene la persistencia de IDs (`track_buffer: 60`) ante oclusiones sin calcular flujo óptico innecesario.
* **OpenCV (`opencv-python`):** Captura de video nativa por DirectShow (`CAP_DSHOW`) para baja latencia en webcam, manejo de buffers FFmpeg en RTSP/HTTP y operaciones morfológicas/geométricas.
* **CustomTkinter & Tkinter:** Interfaz gráfica moderna en modo oscuro, enriquecida con controles DVR, visualizador de imágenes con soporte de zoom digital y panel de administración forense.
* **SQLite 3:** Almacenamiento local ligero en modo WAL (Write-Ahead Logging) para registrar eventos sin generar bloqueos en la lectura de video.

---

## 🏗️ Arquitectura del Sistema (Pipeline Desacoplado)

El sistema opera mediante 4 hilos concurrentes completamente independientes:

```text
[ CaptureThread ]  ---> Cola (maxsize=1, latest-frame-wins)
       │
       v
[ InferWorker ]    ---> YOLOv8 + ByteTrack + Analítica de cruce y zonas
       │
       ├──> [ EventWriter ]  ---> Escritura JPEG + INSERT SQLite (Asíncrono)
       │
       v
[ GUI Mainloop ]   ---> Renderizado constante a 30 FPS vía after() (Thread-Safe)
CaptureThread: Lee la fuente de video a velocidad nativa, descarta cuadros acumulados y gestiona reconexión automática en streams IP.

InferWorker: Procesa la analítica espacial sobre una resolución fija de 960x540 píxeles.

EventWriter: Guarda la evidencia fotográfica limpia y registra la infracción en SQLite sin demorar el video en vivo.

GUI Mainloop: Recibe los fotogramas anotados y actualiza la pantalla a 30 FPS mediante llamadas seguras en Tkinter.

🚀 Funcionalidades Principales
Monitoreo en Tiempo Real:

Soporte para Webcam integrada (con corrección de modo espejo).

Soporte para cámara de celular / stream IP (HTTP MJPEG / RTSP de baja latencia).

Reproductor de video local (MP4) con funciones DVR: pausa/play, rebobinado, adelanto ±5s y selector de velocidad (0.5x, 1.0x, 1.5x, 2.0x).

Calibración Interactiva en Pantalla:

Línea de Conteo (2 clics): Registra Entradas y Salidas con cálculo vectorial y banda de histéresis para evitar duplicaciones.

Línea de Cruce Prohibido (2 clics): Alerta instantánea si se traspasa el límite.

Zona Restringida (4 clics): Detección de intrusión por inclusión de polígono (pointPolygonTest).

Auditoría Forense Integrada:

Pestaña de galería con vista en cuadrícula de todas las capturas registradas.

Visor modal interactivo con zoom mediante rueda del mouse y navegación por flechas (←/→).

Exportación de la base de datos a reportes en formato CSV.

📦 Instalación y Uso
1. Clonar el repositorio
Bash
git clone [https://github.com/HernanJairo/cctv-reconocimiento-visual.git](https://github.com/HernanJairo/cctv-reconocimiento-visual.git)
cd cctv-reconocimiento-visual
2. Instalar dependencias
Bash
pip install -r requirements.txt
3. Ejecutar
Bash
python main.py
Nota: Al iniciar, la aplicación realiza un calentamiento del modelo en memoria (~5 segundos) para asegurar estabilidad de ejecución en Windows. Una vez completado, el botón "Iniciar" se habilitará automáticamente.