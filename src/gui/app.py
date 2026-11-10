"""Aplicación principal — Sistema Integral de Videovigilancia (GUI).

Regla de oro de la arquitectura: Tkinter SOLO se toca desde el hilo del
mainloop. Los workers (captura / inferencia / escritura) publican datos en
slots y colas; la GUI los drena con after() a frecuencia fija.
"""
import os
import queue
import time
import tkinter as tk
import tkinter.filedialog as fd
from datetime import datetime

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

from ..capture import CaptureThread, FrameSlot
from ..config import Settings
from ..engine import InferWorker
from ..geometry import Calibration
from ..paths import VIDEOS_DIR
from ..storage import EventWriter, export_csv, fetch_events, init_db
from .viewer import ImageViewerModal

SOURCES = ["Webcam Integrada", "Celular (IP/DroidCam)", "Archivo de Video (MP4)"]
SPEEDS = ["0.5x", "1.0x", "1.5x", "2.0x"]

# ---- Paleta VMS / CCTV comercial ----
C_BG       = "#0b0f19"   # fondo raíz (grafito profundo)
C_PANEL    = "#0f172a"   # paneles / topbar
C_CARD     = "#1e293b"   # tarjetas y botones secundarios
C_BORDER   = "#334155"   # bordes sutiles
C_TEXT     = "#e2e8f0"
C_MUTED    = "#94a3b8"
C_ACCENT   = "#2563eb"   # acción primaria
C_ACCENT_H = "#1d4ed8"
C_CYAN     = "#06b6d4"   # aforo
C_EMERALD  = "#10b981"   # entradas / iniciar
C_EMERALD_H= "#059669"
C_AMBER    = "#f59e0b"   # salidas / prohibido
C_AMBER_H  = "#d97706"
C_RED      = "#f43f5e"   # alertas / detener
C_RED_H    = "#e11d48"
C_RED_DIM  = "#5b1420"   # pulso LIVE (mitad del ciclo)
C_SCREEN   = "#020617"   # fondo del visor de video


class SecurityApp(ctk.CTk):
    def __init__(self, settings: Settings, model=None):
        super().__init__()
        self.s = settings
        self.configure(fg_color=C_BG)
        self.title("Sistema Integral de Videovigilancia y Auditoría Forense")
        try:
            self.state("zoomed")
        except Exception:
            self.geometry("1440x900")

        init_db()

        # --- pipeline ---
        # El InferWorker vive toda la sesión: el modelo se carga en segundo
        # plano apenas abre la app, así el botón Iniciar responde al instante.
        self.calibration = Calibration()
        self.in_slot = FrameSlot()
        self.out_slot = FrameSlot()
        self.status_q = queue.Queue(maxsize=512)
        self.writer = EventWriter(jpeg_quality=self.s.jpeg_quality)
        self.writer.start()
        self.infer_thread = InferWorker(
            self.s, self.in_slot, self.out_slot, self.calibration,
            self.writer, self.status_q, model=model)
        self.infer_thread.start()
        self.cap_thread = None
        self.running = False

        # --- estado de render ---
        self._last_out_seq = -1
        self._cached_pil = None          # PIL del último frame anotado
        self._painted_key = None         # (seq, w, h) ya pintado en pantalla
        self._disp = (0, 0, 1.0, 16, 9)  # ox, oy, scale, img_w, img_h
        self._slider_updating = False
        self._is_seeking = False         # drag del slider DVR en curso
        self._is_file = False
        self._paused = False
        self._dvr_total = 0              # total de frames del archivo
        self._dvr_fps = 30.0
        self._alert_count = 0
        self._live_pulse = False

        self._build_ui()
        self._drain_status()             # loop de mensajes (after)
        self._render_tick()              # loop de render (after)
        self._tick_topbar()              # reloj + pulso LIVE (after)

    # ==================== UI ====================

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.tabs = ctk.CTkTabview(
            self, corner_radius=10, fg_color=C_PANEL,
            segmented_button_fg_color=C_CARD,
            segmented_button_selected_color=C_ACCENT,
            segmented_button_selected_hover_color=C_ACCENT_H,
            segmented_button_unselected_color=C_CARD,
            segmented_button_unselected_hover_color=C_BORDER,
            text_color=C_TEXT, anchor="w")
        self.tabs.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.tab_live = self.tabs.add("  📹  MONITOREO EN VIVO  ")
        self.tab_gallery = self.tabs.add("  📂  EVIDENCIAS / AUDITORÍA  ")

        self._build_live_tab()
        self._build_gallery_tab()

    def _build_live_tab(self):
        self.tab_live.grid_columnconfigure(1, weight=1)
        self.tab_live.grid_rowconfigure(1, weight=1)

        # ==================== TOPBAR VMS ====================
        top = ctk.CTkFrame(self.tab_live, fg_color=C_PANEL, corner_radius=10,
                           border_width=1, border_color=C_BORDER, height=56)
        top.grid(row=0, column=0, columnspan=2, sticky="ew",
                 padx=4, pady=(4, 8))
        top.grid_propagate(False)

        self.lbl_live = ctk.CTkLabel(
            top, text="● OFFLINE",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=C_MUTED)
        self.lbl_live.pack(side="left", padx=(16, 6))
        ctk.CTkLabel(top, text="CAM-01  |  Canal Principal",
                     font=ctk.CTkFont(size=12), text_color=C_MUTED).pack(
                         side="left")

        self.lbl_clock = ctk.CTkLabel(
            top, text="--:--:--",
            font=ctk.CTkFont(family="Consolas", size=19, weight="bold"),
            text_color=C_CYAN)
        self.lbl_clock.pack(side="right", padx=16)

        srcf = ctk.CTkFrame(top, fg_color="transparent")
        srcf.pack(side="right", padx=8)
        ctk.CTkLabel(srcf, text="FUENTE",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=C_MUTED).pack(side="left", padx=(0, 6))
        self.combo_source = ctk.CTkComboBox(
            srcf, values=SOURCES, width=185, height=30, corner_radius=8,
            fg_color=C_CARD, border_color=C_BORDER, button_color=C_ACCENT,
            button_hover_color=C_ACCENT_H, dropdown_fg_color=C_CARD,
            dropdown_hover_color=C_BORDER,
            command=self._on_source_change)
        self.combo_source.pack(side="left")
        self.combo_source.set("Webcam Integrada")
        self.entry_src = ctk.CTkEntry(
            srcf, width=190, height=30, corner_radius=8, fg_color=C_CARD,
            border_color=C_BORDER, text_color=C_TEXT,
            placeholder_text="Índice o URL")
        self.entry_src.pack(side="left", padx=6)
        self.entry_src.insert(0, str(self.s.webcam_index))
        self.btn_browse = ctk.CTkButton(
            srcf, text="📁", width=34, height=30, corner_radius=8,
            fg_color=C_CARD, border_width=1, border_color=C_BORDER,
            hover_color=C_BORDER, command=self._browse_file)
        self.btn_browse.pack(side="left")

        # ==================== SIDEBAR ====================
        sb = ctk.CTkFrame(self.tab_live, width=285, corner_radius=10,
                          fg_color=C_PANEL, border_width=1,
                          border_color=C_BORDER)
        sb.grid(row=1, column=0, sticky="nsew", padx=(4, 8), pady=(0, 4))
        self.sidebar = sb

        ctk.CTkLabel(sb, text="PANEL DE CONTROL",
                     font=ctk.CTkFont(size=17, weight="bold"),
                     text_color=C_TEXT).pack(pady=(14, 0))
        ctk.CTkLabel(sb, text="Analítica en tiempo real",
                     font=ctk.CTkFont(size=11),
                     text_color=C_MUTED).pack(pady=(0, 6))

        # Arranca deshabilitado hasta que el motor termine de cargar el
        # modelo y su warm-up de inferencia (evita la race torch↔FFmpeg).
        self.btn_iniciar = ctk.CTkButton(
            sb, text="⏳  Cargando modelo…", corner_radius=8, height=38,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#475569", hover_color=C_BORDER,
            command=self._toggle_stream, state="disabled")
        self.btn_iniciar.pack(fill="x", padx=15, pady=(8, 4))

        self.lbl_status = ctk.CTkLabel(sb, text="● Detenido",
                                       text_color=C_MUTED, anchor="w")
        self.lbl_status.pack(fill="x", padx=18, pady=(0, 2))

        ctk.CTkLabel(sb, text="CALIBRACIÓN ANALÍTICA",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=C_MUTED, anchor="w").pack(
                         fill="x", padx=16, pady=(10, 4))

        self._mk_btn(sb, "＋  Línea de Conteo  (2 clics)",
                     C_ACCENT, C_ACCENT_H,
                     lambda: self._set_draw_mode("linea_conteo"))
        self._mk_btn(sb, "⛔  Línea Prohibida  (2 clics)",
                     C_AMBER, C_AMBER_H,
                     lambda: self._set_draw_mode("linea_alarma"))
        self._mk_btn(sb, "⬠  Zona Restringida  (4 clics)",
                     C_RED, C_RED_H,
                     lambda: self._set_draw_mode("zona"))
        self._mk_btn(sb, "⟲  Limpiar Calibración",
                     "#475569", C_BORDER, self._clear_calibration)

        ctk.CTkLabel(sb, text="MÉTRICAS EN VIVO",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=C_MUTED, anchor="w").pack(
                         fill="x", padx=16, pady=(12, 4))

        self.lbl_aforo = self._metric_card(sb, "AFORO EN ESCENA",
                                           C_CYAN, big=True)
        self.lbl_aforo.master.pack(fill="x", padx=15, pady=3)

        duo = ctk.CTkFrame(sb, fg_color="transparent")
        duo.pack(fill="x", padx=15, pady=3)
        duo.grid_columnconfigure((0, 1), weight=1)
        self.lbl_in = self._metric_card(duo, "ENTRADAS", C_EMERALD)
        self.lbl_in.master.grid(row=0, column=0, sticky="nsew",
                                padx=(0, 4))
        self.lbl_out = self._metric_card(duo, "SALIDAS", C_AMBER)
        self.lbl_out.master.grid(row=0, column=1, sticky="nsew",
                                 padx=(4, 0))

        self.lbl_alertas = self._metric_card(sb, "ALERTAS", C_RED)
        self.lbl_alertas.master.pack(fill="x", padx=15, pady=3)

        self.lbl_fps = ctk.CTkLabel(sb, text="Analítica: -- FPS",
                                    font=ctk.CTkFont(size=11),
                                    text_color=C_MUTED)
        self.lbl_fps.pack(pady=(6, 0))

        ctk.CTkButton(sb, text="⬇  Exportar Auditoría CSV",
                      corner_radius=8, height=34,
                      font=ctk.CTkFont(size=12, weight="bold"),
                      fg_color=C_ACCENT, hover_color=C_ACCENT_H,
                      command=self._export_csv).pack(fill="x", padx=15,
                                                     pady=(12, 10))

        # ==================== ÁREA PRINCIPAL ====================
        mc = ctk.CTkFrame(self.tab_live, fg_color=C_BG,
                          corner_radius=10)
        mc.grid(row=1, column=1, sticky="nsew", padx=(0, 4), pady=(0, 4))
        mc.grid_rowconfigure(0, weight=1)
        mc.grid_columnconfigure(0, weight=1)

        # tk.Label plano (no CTkLabel): la imagen se pinta 1:1 en píxeles
        # reales y event.x/y son relativos al mismo widget que la contiene.
        # Con CTkImage la imagen renderiza a size*widget_scaling (HiDPI) y
        # los clics caen sobre un label interno con otro origen → desfase.
        self.video_label = tk.Label(
            mc, text="Seleccione fuente y presione 'Iniciar'",
            bg=C_SCREEN, fg=C_MUTED, font=("Segoe UI", 13))
        self.video_label.grid(row=0, column=0, sticky="nsew",
                              padx=5, pady=5)
        self.video_label.bind("<Button-1>", self._on_canvas_click)
        self.video_label.bind("<Configure>", lambda e: self._repaint())

        # ==================== DVR ====================
        dvr = ctk.CTkFrame(mc, fg_color=C_PANEL, corner_radius=10,
                           border_width=1, border_color=C_BORDER)
        dvr.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))

        self.timeline = ctk.CTkSlider(
            dvr, from_=0, to=100, height=18,
            fg_color="#0b1220", progress_color=C_ACCENT,
            button_color=C_CYAN, button_hover_color="#22d3ee",
            command=self._on_slider_move)
        self.timeline.pack(fill="x", padx=15, pady=(10, 4))
        self.timeline.set(0)
        # CTkSlider.bind reenvía al canvas interno con add="+": los
        # callbacks propios del widget siguen funcionando. Press congela
        # el slider; Release envía un único seek_abs al frame destino.
        self.timeline.bind("<ButtonPress-1>", self._on_seek_press)
        self.timeline.bind("<ButtonRelease-1>", self._on_seek_release)

        ctrl = ctk.CTkFrame(dvr, fg_color="transparent")
        ctrl.pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkButton(ctrl, text="⏪  5s", width=58, height=30,
                      corner_radius=8, fg_color=C_CARD, border_width=1,
                      border_color=C_BORDER, hover_color=C_BORDER,
                      font=ctk.CTkFont(size=12, weight="bold"),
                      command=lambda: self._send_ctl("seek_rel", -5)
                      ).pack(side="left", padx=(0, 4))
        self.btn_pause = ctk.CTkButton(
            ctrl, text="⏸", width=52, height=30, corner_radius=8,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=C_ACCENT, hover_color=C_ACCENT_H,
            command=self._toggle_pause)
        self.btn_pause.pack(side="left", padx=4)
        ctk.CTkButton(ctrl, text="5s  ⏩", width=58, height=30,
                      corner_radius=8, fg_color=C_CARD, border_width=1,
                      border_color=C_BORDER, hover_color=C_BORDER,
                      font=ctk.CTkFont(size=12, weight="bold"),
                      command=lambda: self._send_ctl("seek_rel", 5)
                      ).pack(side="left", padx=(4, 0))

        self.seg_speed = ctk.CTkSegmentedButton(
            ctrl, values=SPEEDS, height=30, corner_radius=15,
            fg_color=C_CARD, selected_color=C_ACCENT,
            selected_hover_color=C_ACCENT_H, unselected_color=C_CARD,
            unselected_hover_color=C_BORDER, text_color=C_TEXT,
            command=self._on_speed_change)
        self.seg_speed.set("1.0x")
        self.seg_speed.pack(side="left", padx=18)

        self.lbl_time = ctk.CTkLabel(
            ctrl, text="00:00 / 00:00",
            font=ctk.CTkFont(family="Consolas", size=13, weight="bold"),
            text_color=C_TEXT)
        self.lbl_time.pack(side="right", padx=6)

        self.log_box = ctk.CTkTextbox(
            mc, height=82, corner_radius=10, fg_color="#080d18",
            border_width=1, border_color=C_BORDER, text_color=C_MUTED,
            font=ctk.CTkFont(family="Consolas", size=11))
        self.log_box.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self._log("Sistema listo. Seleccione la cámara o video para iniciar.")

    # ---- helpers de construcción ----

    @staticmethod
    def _mk_btn(parent, text, fg, hover, cmd):
        b = ctk.CTkButton(parent, text=text, corner_radius=8, height=34,
                          fg_color=fg, hover_color=hover, anchor="w",
                          font=ctk.CTkFont(size=12, weight="bold"),
                          command=cmd)
        b.pack(fill="x", padx=15, pady=3)
        return b

    @staticmethod
    def _metric_card(parent, title, color, big=False):
        """Tarjeta de métrica dashboard; devuelve el label del valor."""
        card = ctk.CTkFrame(parent, fg_color=C_CARD, corner_radius=10,
                            border_width=1, border_color=C_BORDER)
        ctk.CTkLabel(card, text=title,
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=C_MUTED).pack(pady=(7, 0))
        lbl = ctk.CTkLabel(card, text="0",
                           font=ctk.CTkFont(size=30 if big else 22,
                                            weight="bold"),
                           text_color=color)
        lbl.pack(pady=(0, 6))
        return lbl

    def _build_gallery_tab(self):
        self.tab_gallery.grid_columnconfigure(0, weight=1)
        self.tab_gallery.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(self.tab_gallery, height=52, fg_color=C_PANEL,
                           corner_radius=10, border_width=1,
                           border_color=C_BORDER)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        ctk.CTkLabel(top, text="Registro Forense de Evidencias Capturadas",
                     font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=C_TEXT).pack(
                         side="left", padx=15, pady=10)
        ctk.CTkButton(top, text="🔄  Actualizar Galería", corner_radius=8,
                      fg_color=C_ACCENT, hover_color=C_ACCENT_H,
                      command=self._load_gallery).pack(side="right", padx=15, pady=10)

        self.gallery_scroll = ctk.CTkScrollableFrame(
            self.tab_gallery, fg_color=C_PANEL,
            label_text="Doble clic en la foto para abrir visor interactivo")
        self.gallery_scroll.grid(row=1, column=0, sticky="nsew",
                                 padx=10, pady=(0, 10))
        self.gallery_scroll.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self._evidencias = []
        self._load_gallery()

    # ==================== FUENTES ====================

    def _on_source_change(self, choice):
        self.entry_src.delete(0, "end")
        if choice == "Webcam Integrada":
            self.entry_src.insert(0, str(self.s.webcam_index))
        elif choice == "Archivo de Video (MP4)":
            self.entry_src.insert(0, self.s.default_video)
        else:
            self.entry_src.insert(0, self.s.default_stream_url)

    def _browse_file(self):
        path = fd.askopenfilename(
            initialdir=VIDEOS_DIR,
            filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov"), ("Todos", "*.*")])
        if path:
            self.combo_source.set("Archivo de Video (MP4)")
            self.entry_src.delete(0, "end")
            self.entry_src.insert(0, path)

    def _build_spec(self):
        sel = self.combo_source.get()
        raw = self.entry_src.get().strip()
        if sel == "Webcam Integrada":
            idx = int(raw) if raw.isdigit() else self.s.webcam_index
            return {"kind": "webcam", "index": idx}
        if sel == "Archivo de Video (MP4)":
            return {"kind": "file", "path": raw}
        return {"kind": "stream", "url": raw}

    def _toggle_stream(self):
        if not self.running:
            spec = self._build_spec()
            if spec["kind"] == "file" and not os.path.exists(spec["path"]):
                self._log(f"Error: archivo no encontrado: {spec['path']}")
                return
            self._start_pipeline(spec)
        else:
            self._stop_pipeline()

    def _start_pipeline(self, spec):
        self._set_status("Conectando…", C_AMBER)
        self.btn_iniciar.configure(text="⏹  Detener Cámara",
                                   fg_color=C_RED, hover_color=C_RED_H)
        # imgsz por fuente: un stream IP infiere con entrada menor para
        # absorber jitter/lag de red sin sacrificar FPS de analítica.
        imgsz = (self.s.imgsz_stream if spec["kind"] == "stream"
                 else self.s.imgsz)
        self.infer_thread.send("set_imgsz", imgsz)
        self.cap_thread = CaptureThread(spec, self.s, self.in_slot, self.status_q)
        self.cap_thread.start()
        self.running = True
        self._paused = False
        self._is_seeking = False
        self.btn_pause.configure(text="⏸", fg_color=C_ACCENT)
        self._log(f"Iniciando pipeline de video… (inferencia imgsz={imgsz})")

    def _stop_pipeline(self):
        self.running = False
        if self.cap_thread:
            self.cap_thread.stop()
            self.cap_thread = None
        self.btn_iniciar.configure(text="▶  Iniciar Cámara / Transmisión",
                                   fg_color=C_EMERALD,
                                   hover_color=C_EMERALD_H)
        self._set_status("Detenido", C_MUTED)
        # Soltar el frame cacheado: sin esto, un <Configure> (resize)
        # repintaría el último cuadro sobre el texto "Cámara detenida".
        self._cached_pil = None
        self._painted_key = None
        self.video_label.configure(image="", text="Cámara detenida")
        self._log("Monitoreo detenido por el usuario.")

    # ==================== LOOPS GUI (after) ====================

    def _drain_status(self):
        """Consume mensajes de los workers. Corre en el mainloop."""
        try:
            while True:
                m = self.status_q.get_nowait()
                self._handle_status(m)
        except queue.Empty:
            pass
        self.after(60, self._drain_status)

    def _handle_status(self, m):
        kind, src = m.get("kind"), m.get("src")
        if kind == "state":
            st = m.get("state")
            colors = {"connecting": C_AMBER, "ok": C_EMERALD,
                      "reconnecting": C_AMBER, "paused": C_CYAN,
                      "error": C_RED, "stopped": C_MUTED,
                      "loading_model": C_AMBER, "model_ready": C_EMERALD}
            labels = {"connecting": "Conectando…", "ok": "En línea",
                      "reconnecting": "Reconectando…", "paused": "Pausado",
                      "error": "Error", "stopped": "Detenido",
                      "loading_model": "Cargando modelo…",
                      "model_ready": "Modelo listo"}
            if st == "ok" and src == "engine":
                pass  # el motor en reposo no es "En línea" de video
            elif st in labels:
                self._set_status(labels[st], colors.get(st, C_MUTED))
            if m.get("msg"):
                self._log(m["msg"])
            if st == "model_ready" and not self.running:
                self.btn_iniciar.configure(
                    text="▶  Iniciar Cámara / Transmisión",
                    fg_color=C_EMERALD, hover_color=C_EMERALD_H,
                    state="normal")
                self._set_status("Listo para iniciar", C_EMERALD)
            if st == "ok" and src == "capture":
                self._is_file = bool(m.get("is_file"))
            if st == "error":
                self._stop_pipeline()
                self._set_status("Error en la fuente", C_RED)
        elif kind == "progress" and self._is_file:
            cur, tot, fps = m["frame"], m["total"], m["fps"] or 30
            self._dvr_total, self._dvr_fps = tot, fps
            # Durante el drag NO se toca el slider ni el reloj: la
            # posición la manda el dedo del usuario, no el playback.
            if not self._is_seeking:
                self._slider_updating = True
                self.timeline.set((cur / tot) * 100)
                self._slider_updating = False
                cs, ts = int(cur / fps), int(tot / fps)
                self.lbl_time.configure(
                    text=f"{cs//60:02d}:{cs%60:02d} / "
                         f"{ts//60:02d}:{ts%60:02d}")
        elif kind == "event":
            self._log(m["msg"])
        elif kind == "alert":
            self._alert_count += 1
            self.lbl_alertas.configure(text=str(self._alert_count))

    def _render_tick(self):
        """Compositor desacoplado a render_fps. Corre en el mainloop."""
        data, seq = self.out_slot.latest()
        if data is not None and seq != self._last_out_seq:
            self._last_out_seq = seq
            stats = data["stats"]
            self.lbl_aforo.configure(text=str(stats["aforo"]))
            self.lbl_in.configure(text=str(stats["in"]))
            self.lbl_out.configure(text=str(stats["out"]))
            self.lbl_fps.configure(
                text=f"Analítica: {stats['infer_fps']:.1f} FPS")

            rgb = cv2.cvtColor(data["frame"], cv2.COLOR_BGR2RGB)
            self._cached_pil = Image.fromarray(rgb)

        # Hold-last-frame: si captura/inferencia se demora con un frame
        # (frame drops del stream), en cada tick se repinta el último
        # cuadro cacheado — la pantalla nunca queda en negro ni alterna
        # imagen/texto, así no hay "flash" ni parpadeo. _repaint() es
        # idempotente: no repite el resize si (seq, tamaño) no cambió.
        if self.running and self._cached_pil is not None:
            self._repaint()

        self.after(max(15, int(1000 / self.s.render_fps)), self._render_tick)

    def _repaint(self):
        if self._cached_pil is None:
            return
        w = max(100, self.video_label.winfo_width())
        h = max(100, self.video_label.winfo_height())
        key = (self._last_out_seq, w, h)
        if key == self._painted_key:
            return                       # mismo cuadro al mismo tamaño
        iw, ih = self._cached_pil.size
        scale = min(w / iw, h / ih)
        dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
        ox, oy = (w - dw) // 2, (h - dh) // 2
        self._disp = (ox, oy, scale, iw, ih)

        # ImageTk.PhotoImage no aplica escalado de widgets: la imagen se
        # muestra exactamente dw×dh px y centrada por el anchor del label,
        # por lo que (ox, oy, scale) describe el letterbox real en pantalla.
        resized = self._cached_pil.resize((dw, dh), Image.Resampling.BILINEAR)
        tk_img = ImageTk.PhotoImage(resized)
        self.video_label.configure(image=tk_img, text="")
        self.video_label.image = tk_img
        self._painted_key = key

    def _tick_topbar(self):
        """Reloj digital + indicador LIVE pulsante. Corre en el mainloop."""
        self.lbl_clock.configure(text=datetime.now().strftime("%H:%M:%S"))
        if self.running:
            self._live_pulse = not self._live_pulse
            self.lbl_live.configure(
                text="● LIVE",
                text_color=C_RED if self._live_pulse else C_RED_DIM)
        else:
            self._live_pulse = False
            self.lbl_live.configure(text="● OFFLINE", text_color=C_MUTED)
        self.after(500, self._tick_topbar)

    # ==================== CALIBRACIÓN ====================

    def _set_draw_mode(self, mode):
        if not self.running:
            self._log("Inicie el video antes de calibrar.")
            return
        self.calibration.set_mode(mode)
        n = {"linea_conteo": 2, "linea_alarma": 2, "zona": 4}[mode]
        self._log(f"Modo {mode.upper()}: haga {n} clics sobre el video.")

    def _clear_calibration(self):
        self.calibration.clear()
        if self.infer_thread:
            self.infer_thread.send("reset_counters")
        self.lbl_in.configure(text="0")
        self.lbl_out.configure(text="0")
        self._alert_count = 0
        self.lbl_alertas.configure(text="0")
        self._log("Calibración y contadores reiniciados.")

    def _on_canvas_click(self, event):
        if not self.running:
            return
        ox, oy, scale, iw, ih = self._disp
        xn = (event.x - ox) / (iw * scale)
        yn = (event.y - oy) / (ih * scale)
        if not (0.0 <= xn <= 1.0 and 0.0 <= yn <= 1.0):
            return
        if self.calibration.add_point(xn, yn):
            self._log("Calibración completada y activa.")

    # ==================== DVR ====================

    def _send_ctl(self, *cmd):
        if self.running and self.cap_thread:
            self.cap_thread.send(*cmd)

    def _toggle_pause(self):
        if not self.running:
            return
        self._paused = not self._paused
        self._send_ctl("pause", self._paused)
        self.btn_pause.configure(
            text="▶" if self._paused else "⏸",
            fg_color=C_EMERALD if self._paused else C_ACCENT)

    def _on_seek_press(self, _event):
        """ButtonPress sobre el slider: congela las actualizaciones de
        progreso para que el thumb siga al mouse y no al playback."""
        self._is_seeking = True

    def _on_seek_release(self, _event):
        """ButtonRelease: envía UN salto absoluto al frame elegido. El
        CaptureThread hace cap.set(CAP_PROP_POS_FRAMES) y purga el buffer
        → la imagen salta de inmediato al segundo exacto."""
        if not self._is_seeking:
            return
        self._is_seeking = False
        if not (self._is_file and self.cap_thread):
            return
        target = int((self.timeline.get() / 100.0)
                     * self.cap_thread.total_frames)
        self._send_ctl("seek_abs", target)

    def _on_slider_move(self, value):
        # El command también se dispara con timeline.set() (progreso) —
        # esos cambios se ignoran por _slider_updating. Durante el drag
        # solo se muestra la previsualización del tiempo objetivo: el
        # seek real se envía una sola vez al soltar el mouse.
        if self._slider_updating or not self._is_file:
            return
        if self._is_seeking:
            fps = self._dvr_fps or 30
            tot = self._dvr_total or 1
            cs, ts = int((value / 100.0) * tot / fps), int(tot / fps)
            self.lbl_time.configure(
                text=f"{cs//60:02d}:{cs%60:02d} / "
                     f"{ts//60:02d}:{ts%60:02d}")

    def _on_speed_change(self, speed_str):
        self._send_ctl("speed", float(speed_str.replace("x", "")))

    # ==================== GALERÍA / CSV ====================

    def _load_gallery(self):
        for w in self.gallery_scroll.winfo_children():
            w.destroy()
        self._evidencias = fetch_events(with_photo_only=True)

        if not self._evidencias:
            ctk.CTkLabel(self.gallery_scroll,
                         text="No hay capturas registradas aún.",
                         font=ctk.CTkFont(size=16)).pack(pady=40)
            return

        for idx, ev in enumerate(self._evidencias):
            ev_id, fecha, tipo, pid, ruta = ev
            row, col = idx // 4, idx % 4
            card = ctk.CTkFrame(self.gallery_scroll, corner_radius=10,
                                fg_color=C_CARD, border_width=1,
                                border_color=C_BORDER)
            card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")

            if os.path.exists(ruta):
                try:
                    pil = Image.open(ruta)
                    pil.thumbnail((220, 130))
                    cimg = ctk.CTkImage(light_image=pil, dark_image=pil,
                                        size=(220, 130))
                    lbl_th = ctk.CTkLabel(card, image=cimg, text="",
                                          cursor="hand2")
                except Exception:
                    lbl_th = ctk.CTkLabel(card, text="[Error]")
            else:
                lbl_th = ctk.CTkLabel(card, text="[No encontrado]")
            lbl_th.pack(padx=8, pady=(8, 4))

            lbl_tipo = ctk.CTkLabel(card, text=f"⚠️ {tipo}",
                                    font=ctk.CTkFont(weight="bold"),
                                    text_color=C_RED)
            lbl_tipo.pack()
            lbl_info = ctk.CTkLabel(card, text=f"ID: #{pid}\n{fecha}",
                                    font=ctk.CTkFont(size=11),
                                    text_color=C_MUTED)
            lbl_info.pack(pady=(0, 6))

            def _open(e, i=idx):
                ImageViewerModal(self, self._evidencias, i)

            for w in (card, lbl_th, lbl_tipo, lbl_info):
                w.bind("<Double-Button-1>", _open)

    def _export_csv(self):
        path = export_csv()
        self._log(f"Reporte exportado: {os.path.basename(path)}")

    # ==================== MISC ====================

    def _log(self, text):
        hora = datetime.now().strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{hora}] {text}\n")
        self.log_box.see("end")
        # Buffer acotado: sin recorte el Textbox crece indefinidamente en
        # sesiones largas con muchos eventos (fuga de memoria en la GUI).
        if int(self.log_box.index("end-1c").split(".")[0]) > 400:
            self.log_box.delete("1.0", "201.0")

    def _set_status(self, text, color):
        self.lbl_status.configure(text=f"● {text}", text_color=color)

    def on_closing(self):
        self.running = False
        if self.cap_thread:
            self.cap_thread.stop()
        self.infer_thread.stop()
        self.writer.stop()
        for t in (self.cap_thread, self.infer_thread):
            if t:
                t.join(timeout=2.0)
        self.writer.join(timeout=2.0)
        self.destroy()
