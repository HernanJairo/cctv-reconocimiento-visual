"""Persistencia asíncrona: WriterThread desacopla I/O de disco del pipeline.

El motor de inferencia encola trabajos (foto de evidencia + evento SQL) y
sigue procesando frames sin bloquearse jamás por cv2.imwrite ni por un
commit de SQLite.
"""
import csv
import os
import queue
import sqlite3
import threading
from datetime import datetime

import cv2

from .paths import DB_PATH, CAPTURAS_DIR, EXPORTS_DIR

_JOB_STOP = object()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eventos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_hora TEXT NOT NULL,
            tipo_evento TEXT NOT NULL,
            persona_id INTEGER,
            ruta_foto TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_eventos_tipo ON eventos(tipo_evento)")
    conn.commit()
    conn.close()


class EventWriter(threading.Thread):
    """Hilo único dueño de la conexión SQLite y de la escritura de fotos.

    Trabajos soportados (tuplas por la cola):
      ("photo", frame_bgr, ruta_abs)           → cv2.imwrite
      ("event", tipo, persona_id, ruta_foto)   → INSERT
      ("alert", tipo, persona_id, frame, ruta) → foto + INSERT atómicos
    """

    def __init__(self, jpeg_quality=90):
        super().__init__(daemon=True, name="EventWriter")
        self.q = queue.Queue(maxsize=256)
        self.jpeg_quality = jpeg_quality
        self._dropped = 0

    # --- API pública (llamada desde el motor) ---
    def enqueue_event(self, tipo, persona_id, ruta_foto=""):
        self._put(("event", tipo, int(persona_id), ruta_foto))

    def enqueue_alert(self, tipo, persona_id, frame_bgr, ruta_foto):
        self._put(("alert", tipo, int(persona_id), frame_bgr, ruta_foto))

    def _put(self, job):
        try:
            self.q.put_nowait(job)
        except queue.Full:
            self._dropped += 1  # nunca bloquear el pipeline por I/O

    def stop(self):
        self._put(_JOB_STOP)

    # --- Loop interno ---
    def run(self):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        cur = conn.cursor()
        pending = 0
        while True:
            try:
                job = self.q.get(timeout=0.5)
            except queue.Empty:
                if pending:
                    conn.commit()
                    pending = 0
                continue
            if job is _JOB_STOP:
                break
            try:
                if job[0] == "photo":
                    _, frame, ruta = job
                    cv2.imwrite(ruta, frame,
                                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                elif job[0] == "event":
                    _, tipo, pid, ruta = job
                    cur.execute(
                        "INSERT INTO eventos (fecha_hora, tipo_evento, persona_id, ruta_foto)"
                        " VALUES (?, ?, ?, ?)",
                        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tipo, pid, ruta))
                    pending += 1
                elif job[0] == "alert":
                    _, tipo, pid, frame, ruta = job
                    os.makedirs(os.path.dirname(ruta), exist_ok=True)
                    cv2.imwrite(ruta, frame,
                                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                    cur.execute(
                        "INSERT INTO eventos (fecha_hora, tipo_evento, persona_id, ruta_foto)"
                        " VALUES (?, ?, ?, ?)",
                        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tipo, pid, ruta))
                    pending += 1
                if pending >= 20:           # commit por lotes
                    conn.commit()
                    pending = 0
            except Exception:
                pass                        # I/O nunca debe matar el writer
        if pending:
            conn.commit()
        conn.close()


# --- Lecturas (galería / exportación) — conexión de solo lectura por consulta ---

def fetch_events(with_photo_only=False, limit=500):
    conn = sqlite3.connect(DB_PATH)
    sql = "SELECT id, fecha_hora, tipo_evento, persona_id, ruta_foto FROM eventos"
    if with_photo_only:
        sql += " WHERE ruta_foto != ''"
    sql += " ORDER BY id DESC LIMIT ?"
    rows = conn.execute(sql, (limit,)).fetchall()
    conn.close()
    return rows


def export_csv():
    rows = fetch_events(with_photo_only=False, limit=1_000_000)
    path = os.path.join(
        EXPORTS_DIR,
        f"reporte_auditoria_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Fecha_Hora", "Tipo_Evento", "Persona_Track_ID", "Ruta_Foto"])
        w.writerows(rows)
    return path


def new_capture_path(track_id, tag="alerta"):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(CAPTURAS_DIR, f"{tag}_ID{track_id}_{ts}.jpg")
