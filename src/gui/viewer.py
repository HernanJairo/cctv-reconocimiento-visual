"""Visor modal de evidencia forense (zoom + navegación)."""
import os

import customtkinter as ctk
from PIL import Image


class ImageViewerModal(ctk.CTkToplevel):
    def __init__(self, parent, lista_evidencias, indice_inicial=0):
        super().__init__(parent)
        self.title("Visor de Evidencia Forense")
        self.geometry("1100x750")
        self.configure(fg_color="#09090b")
        self.transient(parent)
        self.grab_set()

        self.evidencias = lista_evidencias
        self.index = indice_inicial
        self.zoom_level = 1.0
        self.current_pil_img = None

        self.top_bar = ctk.CTkFrame(self, fg_color="#18181b", height=45)
        self.top_bar.pack(fill="x", padx=10, pady=(10, 0))

        self.lbl_title = ctk.CTkLabel(
            self.top_bar, text="Detalle de Evidencia",
            font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_title.pack(side="left", padx=15)

        self.btn_close = ctk.CTkButton(
            self.top_bar, text="Cerrar (Esc)", width=90,
            fg_color="#ef4444", hover_color="#dc2626", command=self.destroy)
        self.btn_close.pack(side="right", padx=10, pady=6)

        self.center_frame = ctk.CTkFrame(self, fg_color="#000000")
        self.center_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.center_frame.bind("<Configure>", lambda e: self.render_image())

        self.lbl_img = ctk.CTkLabel(self.center_frame, text="")
        self.lbl_img.pack(fill="both", expand=True)

        self.bottom_bar = ctk.CTkFrame(self, fg_color="#18181b", height=50)
        self.bottom_bar.pack(fill="x", padx=10, pady=(0, 10))

        self.lbl_info = ctk.CTkLabel(self.bottom_bar, text="",
                                     font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_info.pack(side="left", padx=20, pady=10)

        self.lbl_zoom = ctk.CTkLabel(
            self.bottom_bar,
            text="Zoom: 100% (Rueda: Zoom | ← →: Navegar)",
            text_color="#9ca3af")
        self.lbl_zoom.pack(side="right", padx=20, pady=10)

        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<Left>", lambda e: self.prev_image())
        self.bind("<Right>", lambda e: self.next_image())
        self.bind("<MouseWheel>", self.on_mouse_wheel)

        self.load_current_image()

    def load_current_image(self):
        if not self.evidencias:
            return
        self.zoom_level = 1.0
        ev_id, fecha, tipo, pid, ruta = self.evidencias[self.index]
        self.lbl_info.configure(
            text=f"Evento: {tipo}  |  Sujeto ID: #{pid}  |  Fecha: {fecha}  "
                 f"({self.index + 1}/{len(self.evidencias)})")
        try:
            self.current_pil_img = Image.open(ruta) if os.path.exists(ruta) else None
        except Exception:
            self.current_pil_img = None
        self.render_image()

    def render_image(self):
        if self.current_pil_img is None:
            self.lbl_img.configure(image="", text="Imagen no disponible")
            return

        w_box = max(200, self.center_frame.winfo_width())
        h_box = max(200, self.center_frame.winfo_height())

        img_w, img_h = self.current_pil_img.size
        ratio = min(w_box / img_w, h_box / img_h) * self.zoom_level
        nw, nh = max(10, int(img_w * ratio)), max(10, int(img_h * ratio))

        resized = self.current_pil_img.resize((nw, nh), Image.Resampling.BILINEAR)
        tk_img = ctk.CTkImage(light_image=resized, dark_image=resized,
                              size=(nw, nh))

        self.lbl_img.configure(image=tk_img, text="")
        self.lbl_img.image = tk_img
        self.lbl_zoom.configure(
            text=f"Zoom: {int(self.zoom_level * 100)}% (Rueda: Zoom | ← →: Navegar)")

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.load_current_image()

    def next_image(self):
        if self.index < len(self.evidencias) - 1:
            self.index += 1
            self.load_current_image()

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.zoom_level = min(3.5, self.zoom_level + 0.15)
        else:
            self.zoom_level = max(0.5, self.zoom_level - 0.15)
        self.render_image()
