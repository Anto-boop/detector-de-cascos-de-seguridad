"""Demo local · Detector de casco de seguridad (YOLO26n) · UBO DSI 2026-S2 · Equipo AMIR.
Uso local: pip install -r requirements.txt ; python app.py   (con modelo_casco_yolo26.pt junto a este archivo). También corre en Hugging Face Spaces."""
import os
# ===== CONFIGURACIÓN =====
EQUIPO = "AMIR"
INTEGRANTES = ["Antonia Moya, Isidora Ramos (D01)"]
URL_DATASET = "https://universe.roboflow.com/roboflow-universe-projects/hard-hats-fhbh5/dataset/6"

# Modelo propio (YOLO26). Si la ruta no existe, la celda 3 te pide subir el archivo .pt.
MODELO_PT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_casco_yolo26.pt")   # deja el .pt junto a app.py
UMBRAL_INICIAL = 0.40     # umbral elegido en VALID para este modelo (máx. F1 "sin casco", notebook Entrenamiento_YOLO26). Si la cámara no detecta nada, bájalo con el slider
IOU_CRUZADO = 0.5        # si casco y sin casco se solapan más que esto, se queda el de mayor confianza
IMGSZ = 640               # tamaño de entrada del modelo (el mismo del entrenamiento)
INTERVALO_S = 2.0         # segundos entre cuadros analizados en la cámara en vivo
VENTANA = 3               # se miran los últimos N cuadros analizados...
MINIMO_POSITIVOS = 2      # ...y la ALERTA se confirma si en al menos M hay 'sin casco'
MAX_SEG_VIDEO = 120       # largo máximo de video a analizar (segundos)
MAX_LLAMADAS_VIDEO = 150  # tope de cuadros analizados por video
GUARDAR_CAPTURAS = True   # guarda una captura por alerta confirmada (contiene imágenes de personas: bórrenlas al terminar)
CARPETA_EVENTOS = "eventos"
USUARIO_CLAVE = (os.environ["DEMO_USUARIO"], os.environ["DEMO_CLAVE"]) if os.environ.get("DEMO_USUARIO") and os.environ.get("DEMO_CLAVE") else None      # p. ej. ("equipo", "una-clave-larga") para pedir usuario y clave al abrir el enlace; None = sin clave

import os, re, io, json, time, shutil, subprocess, tempfile, warnings
from datetime import datetime
from collections import deque
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
import cv2
import gradio as gr
from ultralytics import YOLO

# En Hugging Face (ZeroGPU) la plataforma exige al menos una función con @spaces.GPU. Esta no se usa:
# el modelo es pequeño y corre en la CPU del Space, así no se gasta la cuota diaria de GPU de quien lo abre.
try:
    import spaces
    @spaces.GPU
    def _zerogpu_requisito():
        return None
except Exception:
    pass
DISPOSITIVO = "cpu" if os.environ.get("SPACE_ID") else None      # None = Ultralytics elige (GPU si hay, si no CPU)
warnings.filterwarnings("ignore")

ruta_modelo = MODELO_PT
assert os.path.exists(ruta_modelo), f"Falta el modelo: {ruta_modelo} (déjalo junto a app.py)"

MODELO_OBJ = YOLO(ruta_modelo)
print("Clases del modelo:", MODELO_OBJ.names)
os.makedirs(CARPETA_EVENTOS, exist_ok=True)

class ErrorDeServicio(Exception):
    pass

def _limpiar(msg):
    msg = str(msg)
    return msg if len(msg) <= 300 else msg[:300] + " …"
print("Listo. Modelo local:", os.path.basename(ruta_modelo), "· sin API key")

def categoria(nombre):
    n = " ".join(re.sub(r"[^a-z0-9]+", " ", str(nombre).lower()).split())
    if re.search(r"\b(no|sin|without)\b.*(helmet|hardhat|hard hat|casco)|nohelmet|nohardhat", n): return "sin_casco"
    if re.search(r"helmet|hardhat|hard hat|casco", n): return "casco"
    return "otra"

COLOR = {"sin_casco": (220, 40, 40), "casco": (46, 160, 67)}
ETIQ = {"sin_casco": "SIN CASCO", "casco": "casco"}

def _a_pil(img):
    if isinstance(img, Image.Image): return img.convert("RGB")
    return Image.fromarray(np.asarray(img)).convert("RGB")

def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def detectar(img, umbral):
    # Una pasada del modelo local. Devuelve solo cascos y 'sin casco'.
    # Con umbral bajo el modelo puede marcar la MISMA cabeza como casco y como sin casco:
    # se conserva solo la de mayor confianza cuando dos cajas de clases distintas se solapan (IoU >= IOU_CRUZADO).
    pil = _a_pil(img)
    try:
        r = MODELO_OBJ.predict(pil, conf=float(umbral), imgsz=IMGSZ, device=DISPOSITIVO, verbose=False)[0]
    except Exception as e:
        raise ErrorDeServicio(_limpiar(f"{type(e).__name__}: {e}")) from None
    cand = []
    for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        cat = categoria(r.names[int(k)])
        if cat != "otra": cand.append({"cat": cat, "conf": float(c), "box": tuple(b)})
    cand.sort(key=lambda d: -d["conf"])
    out = []
    for d in cand:
        if any(o["cat"] != d["cat"] and _iou(o["box"], d["box"]) >= IOU_CRUZADO for o in out):
            continue
        out.append(d)
    return out

def estado_cuadro(dets):
    sin = [d for d in dets if d["cat"] == "sin_casco"]
    if sin: return "SIN_CASCO", len(sin), max(d["conf"] for d in sin)
    return ("OK" if dets else "NADA"), 0, 0.0

def anotar(img, dets, sello=None):
    im = _a_pil(img).copy(); d = ImageDraw.Draw(im)
    try: f = ImageFont.load_default(size=max(12, im.width // 50))
    except Exception: f = ImageFont.load_default()
    for x in dets:
        c = COLOR[x["cat"]]; b = x["box"]
        d.rectangle(b, outline=c, width=max(2, im.width // 250))
        t = f"{ETIQ[x['cat']]} {x['conf']:.2f}"; tb = d.textbbox((b[0], b[1]), t, font=f)
        d.rectangle((tb[0]-2, tb[1]-2, tb[2]+2, tb[3]+2), fill=c)
        d.text((b[0], b[1]), t, fill=(255, 255, 255), font=f)
    if sello:
        alto = d.textbbox((0, 0), sello, font=f)[3] + 8
        d.rectangle((0, 0, im.width, alto), fill=(0, 0, 0)); d.text((6, 4), sello, fill=(255, 255, 255), font=f)
    return im

class Monitor:
    # Memoria de los últimos cuadros: confirma la alerta solo si 'sin casco' persiste y registra eventos.
    def __init__(self, ventana=None, minimo=None):
        self.hist = deque(maxlen=ventana or VENTANA); self.minimo = minimo or MINIMO_POSITIVOS
        self.activa = False; self.eventos = []; self.analizados = 0; self.errores = 0; self.con_sin_casco = 0

    def actualizar(self, estado, n_sin, conf_max, cuando, imagen=None):
        self.analizados += 1; hay = estado == "SIN_CASCO"; self.con_sin_casco += hay; self.hist.append(hay)
        persistente = sum(self.hist) >= self.minimo; nuevo = False
        if persistente and not self.activa:
            self.activa = True; nuevo = True
            ev = {"n": len(self.eventos) + 1, "cuando": cuando, "personas_sin_casco": n_sin, "confianza_max": round(conf_max, 3), "captura": ""}
            if GUARDAR_CAPTURAS and imagen is not None:
                ruta = os.path.join(CARPETA_EVENTOS, f"evento_{datetime.now():%Y%m%d_%H%M%S}_{ev['n']}.jpg"); imagen.save(ruta); ev["captura"] = ruta
            self.eventos.append(ev)
        elif self.activa and not any(self.hist):
            self.activa = False
        return persistente, nuevo

    def nivel(self, estado):
        if self.activa: return "ALERTA"
        if estado == "SIN_CASCO": return "POSIBLE"
        return "OK" if estado == "OK" else "NADA"

    def tabla(self):
        cols = ["n", "cuando", "personas_sin_casco", "confianza_max", "captura"]
        return pd.DataFrame(self.eventos, columns=cols)

TARJETAS = {
    "ALERTA":  ("#c62828", "🔴 ALERTA: persona sin casco", "Se confirmó en varios cuadros seguidos. Revisar de inmediato."),
    "POSIBLE": ("#f9a825", "🟡 POSIBLE: se vio 'sin casco' en este cuadro", "Aún no se repite en los cuadros siguientes; se confirma si persiste."),
    "OK":      ("#2e7d32", "🟢 Sin alertas", "Solo se detectaron cascos. Esto no garantiza el cumplimiento de EPP."),
    "NADA":    ("#607d8b", "⚪ Sin detecciones", "No se detectó casco ni 'sin casco' (puede que no haya personas o que estén lejos)."),
    "ERROR":   ("#6a1b9a", "🟣 Error", ""),
    "ESPERA":  ("#607d8b", "⚪ Esperando cámara…", "Permite el acceso a la cámara en el navegador."),
}
def tarjeta(nivel, extra="", detalle=None):
    color, titulo, msg = TARJETAS[nivel]
    from html import escape
    return (f'<div style="border-left:8px solid {color};background:#f6f7f9;color:#1b1f23;padding:10px 14px;border-radius:8px;font-family:system-ui,sans-serif">'
            f'<div style="font-size:18px;font-weight:700;color:{color}">{escape(titulo)}</div>'
            f'<div style="font-size:13.5px;margin-top:3px">{escape(detalle or msg)}</div>'
            f'<div style="font-size:12px;opacity:.75;margin-top:6px">{escape(extra)}</div></div>')

def en_vivo(img, umbral, mon):
    if img is None:
        return None, tarjeta("ESPERA"), (mon.tabla() if mon else pd.DataFrame()), mon
    mon = mon or Monitor()
    ahora = datetime.now().strftime("%H:%M:%S")
    try:
        dets = detectar(img, umbral)
    except ErrorDeServicio as e:
        mon.errores += 1
        return _a_pil(img), tarjeta("ERROR", f"{ahora} · errores acumulados: {mon.errores}", f"No se pudo analizar el cuadro: {e}"), mon.tabla(), mon
    est, n_sin, cmax = estado_cuadro(dets)
    pil = anotar(img, dets, f"{ahora}")
    mon.actualizar(est, n_sin, cmax, ahora, pil)
    nivel = mon.nivel(est)
    extra = f"{ahora} · cuadros analizados: {mon.analizados} · con 'sin casco': {mon.con_sin_casco} · eventos: {len(mon.eventos)} · detecciones en este cuadro: {len(dets)} · umbral {float(umbral):.2f}"
    return pil, tarjeta(nivel, extra), mon.tabla(), mon

def exportar_eventos(mon):
    if not mon or not mon.eventos: return None
    ruta = "eventos_camara.csv"; mon.tabla().to_csv(ruta, index=False); return ruta

def _mmss(seg): return f"{int(seg)//60:02d}:{int(seg)%60:02d}"

def procesar_video(ruta, umbral, cada_s, progreso=gr.Progress()):
    if not ruta or not os.path.exists(ruta):
        return None, tarjeta("ERROR", "", "Primero sube o graba un video."), pd.DataFrame(), None
    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        return None, tarjeta("ERROR", "", "No se pudo abrir el video (formato no soportado o archivo dañado)."), pd.DataFrame(), None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        return None, tarjeta("ERROR", "", "El video no tiene cuadros legibles."), pd.DataFrame(), None
    limite = min(total, int(MAX_SEG_VIDEO * fps)); recortado = total > limite
    paso = max(1, round(fps * float(cada_s)))
    paso = max(paso, int(np.ceil(limite / MAX_LLAMADAS_VIDEO)))          # tope de consultas
    W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    esc = min(1.0, 960 / max(W0, 1)); W, H = int(W0 * esc) // 2 * 2, int(H0 * esc) // 2 * 2
    tmp = tempfile.mkdtemp(); crudo = os.path.join(tmp, "crudo.mp4"); final = os.path.join(tmp, "anotado.mp4")
    wr = cv2.VideoWriter(crudo, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    mon = Monitor(); dets = []; nivel = "NADA"; llamadas = 0; err = None; cons_err = 0
    for i in range(limite):
        ok, bgr = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(cv2.resize(bgr, (W, H)), cv2.COLOR_BGR2RGB)
        if i % paso == 0:
            progreso(i / limite, desc=f"Analizando {_mmss(i/fps)} / {_mmss(limite/fps)}")
            try:
                dets = detectar(rgb, umbral); llamadas += 1; cons_err = 0
                est, n_sin, cmax = estado_cuadro(dets)
                pil = anotar(rgb, dets)
                mon.actualizar(est, n_sin, cmax, _mmss(i / fps), pil); nivel = mon.nivel(est)
            except ErrorDeServicio as e:
                err = str(e); cons_err += 1; mon.errores += 1
                if cons_err >= 3: break                                 # fallos repetidos: no seguir
        wr.write(cv2.cvtColor(np.asarray(anotar(rgb, dets, f"{_mmss(i/fps)}  ·  {TARJETAS[nivel][1].split(':')[0][2:]}")), cv2.COLOR_RGB2BGR))
    cap.release(); wr.release()
    salida = crudo
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", crudo, "-vcodec", "libx264", "-pix_fmt", "yuv420p", final], check=True, timeout=600)
        salida = final
    except Exception:
        pass                                                           # sin ffmpeg: queda el mp4 original (puede no verse en el navegador)
    ev = mon.tabla(); csv = None
    if len(ev): csv = "eventos_video.csv"; ev.to_csv(csv, index=False)
    if llamadas == 0 and err:
        return None, tarjeta("ERROR", "", f"No se pudo analizar el video: {err}"), ev, None
    notas = []
    if recortado: notas.append(f"se analizaron solo los primeros {MAX_SEG_VIDEO} s")
    if paso > round(fps * float(cada_s)): notas.append("se espació el análisis para respetar el tope de cuadros")
    if err: notas.append("hubo errores en algunos cuadros")
    extra = (f"cuadros analizados: {mon.analizados} (1 cada {paso/fps:.1f} s) · con 'sin casco': {mon.con_sin_casco} · eventos confirmados: {len(ev)}"
             + (" · " + "; ".join(notas) if notas else ""))
    nivel_final = "ALERTA" if len(ev) else ("OK" if mon.analizados else "NADA")
    detalle = (f"Se confirmaron {len(ev)} evento(s) de persona sin casco. Ver minuto en la tabla." if len(ev)
               else "No se confirmó ninguna persona sin casco en los cuadros analizados (no garantiza cumplimiento).")
    return salida, tarjeta(nivel_final, extra, detalle), ev, csv

ACERCA = f"""
### Qué es
Demostrador del Proyecto Integrador (DSI · UBO 2026 S2) · **{EQUIPO}** · {", ".join(INTEGRANTES)}.
Detecta **casco / sin casco** con un modelo **YOLO26 propio**, entrenado por el equipo con el dataset [Hard Hats v6]({URL_DATASET}) (Roboflow Universe Projects, CC BY 4.0). **Este modo no usa API key.**

### Cómo decide
1. Se analiza un cuadro cada pocos segundos (no video continuo).
2. Un cuadro con 'sin casco' es **POSIBLE**. Se confirma como **ALERTA** si aparece en al menos {MINIMO_POSITIVOS} de los últimos {VENTANA} cuadros analizados.
3. Cada alerta confirmada queda en la tabla de eventos (hora o minuto del video) con una captura.

### Límites (importante)
- **No es tiempo real continuo:** se analiza un cuadro cada {INTERVALO_S:g} s; se pueden perder eventos más breves. La velocidad depende de si el computador tiene GPU.
- Puede fallar con poca luz, personas lejanas, cascos tapados, ángulos raros o cascos de colores poco comunes.
- No reconoce otros EPP (chaleco, guantes, calzado) ni identifica personas.
- Resultado **orientativo**: apoya la inspección, no la reemplaza; una persona debe confirmar cada alerta.

### Privacidad y seguridad
- Las imágenes se procesan en este computador y no se envían a otros servicios. Usar solo con autorización de las personas filmadas y sin buscar identificarlas.
- Las capturas de alertas se guardan en la carpeta `{CARPETA_EVENTOS}` de este computador: bórrenlas al terminar.
- El enlace público es temporal; usen `USUARIO_CLAVE` si lo comparten.
"""

with gr.Blocks(title="Detector de casco · DSI UBO") as demo:
    gr.Markdown("# 🦺 Detector de casco de seguridad\nApoyo a la inspección en obra · resultado orientativo")
    with gr.Tab("📷 Cámara en vivo"):
        with gr.Row():
            with gr.Column(scale=1):
                cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Cámara")
                umbral_v = gr.Slider(0.05, 0.95, value=UMBRAL_INICIAL, step=0.05, label="Umbral de confianza",
                                     info="Bajo: más detecciones y más falsas alarmas. Alto: menos falsas alarmas, pero puede omitir personas.")
                gr.Markdown(f"Se analiza un cuadro cada **{INTERVALO_S:g} s**.")
            with gr.Column(scale=1):
                salida_v = gr.Image(label="Resultado (rojo: sin casco · verde: casco)")
                tarj_v = gr.HTML(tarjeta("ESPERA"))
        tabla_v = gr.Dataframe(label="Eventos confirmados (cámara)", interactive=False)
        with gr.Row():
            btn_csv_v = gr.Button("Preparar CSV de eventos"); archivo_v = gr.File(label="CSV")
        mon_v = gr.State(None)
        flujo = dict(fn=en_vivo, inputs=[cam, umbral_v, mon_v], outputs=[salida_v, tarj_v, tabla_v, mon_v], concurrency_limit=1)
        try:
            cam.stream(**flujo, stream_every=INTERVALO_S, time_limit=3600)
        except TypeError:
            cam.stream(**flujo, stream_every=INTERVALO_S)
        btn_csv_v.click(exportar_eventos, [mon_v], [archivo_v])
    with gr.Tab("🎞️ Video grabado"):
        with gr.Row():
            with gr.Column(scale=1):
                vid_in = gr.Video(label="Video de la obra (subir o grabar)", sources=["upload", "webcam"])
                umbral_vid = gr.Slider(0.05, 0.95, value=UMBRAL_INICIAL, step=0.05, label="Umbral de confianza")
                cada = gr.Slider(0.5, 10, value=2, step=0.5, label="Analizar un cuadro cada (segundos)",
                                 info=f"Máx. {MAX_SEG_VIDEO} s de video y {MAX_LLAMADAS_VIDEO} cuadros analizados por video.")
                btn = gr.Button("Analizar video", variant="primary")
            with gr.Column(scale=1):
                vid_out = gr.Video(label="Video anotado")
                tarj_vid = gr.HTML(tarjeta("ESPERA", "", "Sube un video y presiona «Analizar video»."))
        tabla_vid = gr.Dataframe(label="Eventos confirmados (video) · «cuando» = minuto:segundo", interactive=False)
        csv_vid = gr.File(label="CSV de eventos del video")
        btn.click(procesar_video, [vid_in, umbral_vid, cada], [vid_out, tarj_vid, tabla_vid, csv_vid], concurrency_limit=1)
    with gr.Tab("ℹ️ Acerca y límites"):
        gr.Markdown(ACERCA)
print("Interfaz construida.")
if __name__ == "__main__":
    demo.queue(max_size=8)
    if os.environ.get("SPACE_ID"):          # Hugging Face Spaces: el servidor debe escuchar en todas las interfaces
        demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), auth=USUARIO_CLAVE)
    else:                                   # computador local: http://127.0.0.1:7860
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True, auth=USUARIO_CLAVE)
