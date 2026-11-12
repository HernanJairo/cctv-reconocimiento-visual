"""Auditoría de streams de prueba.

Prueba cada URL candidata con cv2.VideoCapture (backend FFmpeg) y los
mismos timeouts que usa la app en producción. Cada URL corre en un hilo
daemon con presupuesto de tiempo propio para que una fuente colgada no
frene el resto de la auditoría.

Uso:  python tests/audit_streams.py
"""
import os
import sys
import threading
import time

# Timeouts de socket/lectura ANTES de crear cualquier VideoCapture:
# un servidor caído debe fallar rápido, no bloquear la prueba.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp"
    "|stimeout;6000000"          # RTSP: handshake máx 6 s
    "|timeout;8000000"           # TCP/HTTP: socket I/O máx 8 s
    "|rw_timeout;8000000"        # lectura/escritura máx 8 s
    "|probesize;500000"
    "|analyzeduration;1000000"
    "|fflags;nobuffer|flags;low_delay")

import cv2  # noqa: E402

CANDIDATAS = [
    # --- presets actuales (reportados caídos, se re-verifican) ---
    ("Wowza BBB RTSP", "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov"),
    ("Hessdalen NO RTSP", "rtsp://freja.hiof.no:1935/rtplive/definst/hessdalen03.stream"),
    ("Axis IPVM demo RTSP", "rtsp://demo:demo@ipvmdemo.dyndns.org:5541/onvif-media/media.amp?profile=profile_1_h264&sessiontimeout=60&streamtype=unicast"),
    ("Péndulo Heidelberg MJPEG", "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg"),
    # --- cámaras web públicas adicionales ---
    ("Teide IAC MJPEG", "http://iris.not.iac.es/axis-cgi/mjpg/video.cgi?resolution=320x240"),
    ("St-Malo FR MJPEG", "http://webcam.st-malo.com/axis-cgi/mjpg/video.cgi?resolution=640x480"),
    # --- VOD con PERSONAS (lo ideal para demo de detección) ---
    ("OpenCV vtest.avi (peatones)", "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"),
    ("Intel people-detection", "https://media.githubusercontent.com/media/intel-iot-devkit/sample-videos/master/people-detection.mp4"),
    ("Intel face-demographics-walk", "https://media.githubusercontent.com/media/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4"),
    ("AIGuys cars.mp4 (tráfico)", "https://media.githubusercontent.com/media/theAIGuysCode/yolov4-custom-functions/master/data/video/cars.mp4"),
    ("Ultralytics decelera_portrait", "https://ultralytics.com/assets/decelera_portrait_min.mov"),
    # --- VOD genérico ultra-estable (último recurso) ---
    ("Google BBB MP4", "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"),
    ("Google ForBiggerBlazes", "https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"),
    ("test-videos BBB 360p", "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"),
]

BUDGET_S = 20.0          # presupuesto por fuente
NEED_FRAMES = 5          # nº de frames para darla por buena


def probe(name, url, results):
    t0 = time.time()
    frames, w, h = 0, 0, 0
    err = ""
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        while time.time() - t0 < BUDGET_S and frames < NEED_FRAMES:
            ok, f = cap.read()
            if ok and f is not None:
                frames += 1
                h, w = f.shape[:2]
            else:
                time.sleep(0.1)
        cap.release()
    except Exception as e:
        err = str(e)[:80]
    ok = frames >= NEED_FRAMES
    results.append((ok, name, url, frames, w, h,
                    round(time.time() - t0, 1), err))
    estado = "OK  " if ok else "FAIL"
    print(f"{estado} {name:34s} frames={frames} {w}x{h} "
          f"{time.time() - t0:5.1f}s {err}", flush=True)


def main():
    hilos = []
    for name, url in CANDIDATAS:
        t = threading.Thread(target=probe, args=(name, url, RESULTS),
                             daemon=True)
        t.start()
        hilos.append(t)
    for t in hilos:
        t.join(timeout=BUDGET_S + 15)
    print("\n==== RESUMEN ====")
    for ok, name, url, fr, w, h, dt, err in RESULTS:
        print(("OK  " if ok else "FAIL"), name, "|", url)


RESULTS = []
main()
