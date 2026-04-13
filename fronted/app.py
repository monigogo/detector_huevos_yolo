import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import time

# Configuración 
st.set_page_config(page_title="Detección de Huevos", layout="wide")

st.title("Detección de Huevos")
st.write("Sube una imagen para detectar y clasificar los huevos.")


def filter_egg_detections(result, frame_shape, conf_th, min_area_pct, max_area_pct, min_ar, max_ar, use_roi, roi_y_min, roi_y_max):

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    frame_h, frame_w = frame_shape[:2]
    frame_area = float(frame_h * frame_w)

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=float)
    clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)

    y_min_px = int(roi_y_min * frame_h)
    y_max_px = int(roi_y_max * frame_h)

    filtered = []
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        area_pct = (bw * bh) / frame_area
        aspect_ratio = bw / bh
        cy = (float(y1) + float(y2)) / 2.0

        if confs[i] < conf_th:
            continue
        if area_pct < min_area_pct or area_pct > max_area_pct:
            continue
        if aspect_ratio < min_ar or aspect_ratio > max_ar:
            continue
        if use_roi and not (y_min_px <= cy <= y_max_px):
            continue

        filtered.append(
            {
                "xyxy": (int(x1), int(y1), int(x2), int(y2)),
                "conf": float(confs[i]),
                "cls": int(clss[i]),
                "id": None if ids[i] is None else int(ids[i]),
            }
        )

    return filtered


def draw_filtered_detections(frame_bgr, detections, class_names):
   
    out = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        conf = det["conf"]
        cls_id = det["cls"]
        track_id = det["id"]

        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cls_name = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else str(cls_id)
        if track_id is None:
            label = f"{cls_name} {conf:.2f}"
        else:
            label = f"id:{track_id} {cls_name} {conf:.2f}"
        cv2.putText(out, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

    return out


@st.cache_resource
def load_model():
    
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "train_runs", "train_fold_1", "weights", "best.pt")
    if not os.path.exists(model_path):
        st.error(f"No se encontró el modelo en: {model_path}. Asegúrate de haber entrenado el modelo.")
        return None
    model = YOLO(model_path)
    return model

model = load_model()


st.sidebar.markdown("### Ajustes del Modelo")
imgsz_inference = st.sidebar.slider(
    "Resolución de inferencia (imgsz)",
    min_value=640,
    max_value=1280,
    value=960,
    step=64,
    help="Mayor resolución mejora detección de huevos lejanos, pero consume más tiempo por frame."
)
conf_threshold_image = st.sidebar.slider(
    "Umbral de Confianza (Imagen)",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="En imagen fija suele funcionar bien con umbral medio."
)
conf_threshold_video = st.sidebar.slider(
    "Umbral de Confianza (Video)",
    min_value=0.1,
    max_value=1.0,
    value=0.3,
    step=0.05,
    help="Para huevos pequeños o lejanos en video, usa un valor bajo (0.2-0.35)."
)
iou_threshold = st.sidebar.slider(
    "Solapamiento (IoU)", 
    min_value=0.1, max_value=1.0, value=0.45, step=0.05,
    help="Ajusta cómo se fusionan las cajas superpuestas. Útil para que no cuente el mismo huevo dos veces en el mismo instante."
)
st.sidebar.markdown("### Filtro anti-falsos positivos")
use_geometry_filter = st.sidebar.checkbox(
    "Activar filtro por forma/tamaño/zona",
    value=True,
    help="Reduce detecciones erróneas descartando objetos fuera del patrón esperado del huevo en la cinta."
)
min_area_percent = st.sidebar.slider(
    "Área mínima (%)",
    min_value=0.01,
    max_value=5.0,
    value=0.08,
    step=0.01,
    help="Descarta objetos demasiado pequeños respecto al frame."
)
max_area_percent = st.sidebar.slider(
    "Área máxima (%)",
    min_value=0.2,
    max_value=30.0,
    value=6.0,
    step=0.1,
    help="Descarta objetos demasiado grandes para ser un huevo."
)
min_aspect_ratio = st.sidebar.slider(
    "Relación ancho/alto mínima",
    min_value=0.3,
    max_value=2.0,
    value=0.55,
    step=0.05,
    help="Huevos suelen verse ovalados, no extremadamente delgados."
)
max_aspect_ratio = st.sidebar.slider(
    "Relación ancho/alto máxima",
    min_value=0.6,
    max_value=3.0,
    value=1.65,
    step=0.05,
    help="Huevos suelen verse ovalados, no extremadamente alargados."
)
use_belt_roi = st.sidebar.checkbox(
    "Restringir a franja vertical de la cinta",
    value=False,
    help="Muy útil cuando hay objetos alrededor que no pertenecen a la cinta."
)
roi_y_min = st.sidebar.slider("ROI vertical inicio (0-1)", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
roi_y_max = st.sidebar.slider("ROI vertical fin (0-1)", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
st.sidebar.markdown("---")


source_type = st.sidebar.radio("Selecciona la fuente", ["Imagen Local", "Video Local", "Video de YouTube"])

if source_type == "Imagen Local":
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and model is not None:
        image = Image.open(uploaded_file)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        st.write("Procesando...")
        
        
        results = model(image_cv, conf=0.1, iou=iou_threshold, imgsz=imgsz_inference)
        filtered = filter_egg_detections(
            results[0],
            image_cv.shape,
            conf_threshold_image,
            min_area_percent / 100.0,
            max_area_percent / 100.0,
            min_aspect_ratio,
            max_aspect_ratio,
            use_geometry_filter and use_belt_roi,
            min(roi_y_min, roi_y_max),
            max(roi_y_min, roi_y_max),
        )
        
    
        class_names = results[0].names if hasattr(results[0], "names") else {0: "huevo"}
        res_plotted = draw_filtered_detections(image_cv, filtered, class_names)
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

     
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Original") 
            st.image(image, use_container_width=True)
            
        with col2:
            st.header("Detección")
            st.image(res_plotted_rgb, use_container_width=True)
            
       
        st.write("### Resultados")
        if len(filtered) == 0:
            st.write("No se detectaron huevos en la imagen.")
        else:
            st.write(f"Se detectaron un total de {len(filtered)} elementos.")

elif source_type == "Video Local":
    st.write("### Detección en Video Local (Cinta Transportadora)")
    uploaded_video = st.file_uploader("Sube un video de tu cinta transportadora...", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video is not None and model is not None:
        import tempfile
        
        st.write("Procesando video en tiempo real...")
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
   
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): 
            fps = 30  
        frame_duration = 1.0 / fps

        st_count = st.empty()  
        
        
        col_izq, col_centro, col_der = st.columns([1, 2, 1])
        with col_centro:
            stframe = st.empty()
        
        st.write(f"🎞️ Procesando a {int(fps)} FPS. Presiona 'Stop' en la esquina superior derecha para detener.")
        
        unique_eggs = set()
        egg_frames_count = {}
        
        while cap.isOpened():
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
                
  
            max_width = 640
            h, w = frame.shape[:2]
            if w > max_width:
                ratio = max_width / float(w)
                frame = cv2.resize(frame, (max_width, int(h * ratio)))
                
    
            results = model.track(
                frame, 
                persist=True, 
                verbose=False, 
                conf=conf_threshold_video,
                iou=iou_threshold,
                imgsz=imgsz_inference
            )

            filtered = filter_egg_detections(
                results[0],
                frame.shape,
                conf_threshold_video,
                min_area_percent / 100.0,
                max_area_percent / 100.0,
                min_aspect_ratio,
                max_aspect_ratio,
                use_geometry_filter and use_belt_roi,
                min(roi_y_min, roi_y_max),
                max(roi_y_min, roi_y_max),
            )

            h = frame.shape[0]
            for d in filtered:
                if d["id"] is not None:
                    cy = (d["xyxy"][1] + d["xyxy"][3]) / 2.0
                    # Rango seguro de conteo (evita los bordes)
                    if h * 0.2 < cy < h * 0.8:
                        # Exigir que el huevo se vea al menos en 3 frames consecutivos o no para confirmar
                        # esto evita fantasmas de 1 frame de la IA
                        egg_id = d["id"]
                        egg_frames_count[egg_id] = egg_frames_count.get(egg_id, 0) + 1
                        if egg_frames_count[egg_id] >= 3:
                            unique_eggs.add(egg_id)
                
   
            st_count.markdown(f"### 🥚 Total de huevos contados: **{len(unique_eggs)}**")
            

            class_names = results[0].names if hasattr(results[0], "names") else {0: "huevo"}
            res_plotted = draw_filtered_detections(frame, filtered, class_names)
            res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
   
            stframe.image(res_plotted_rgb, channels="RGB")
            

            elapsed_time = time.time() - start_time
            sleep_time = frame_duration - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            
        cap.release()
        
elif source_type == "Video de YouTube":
    st.write("### Detección en Video de YouTube")
    youtube_url = st.text_input("Ingresa la URL del video de YouTube:")
    
    if st.button("Procesar Video") and youtube_url:
        if model is not None:
            try:
                import yt_dlp
                st.write("Extrayendo información del video...")
                

                # Configurar yt-dlp
                ydl_opts = {
                    'format': 'best[ext=mp4]', 
                    'quiet': True,
                    'nocheckcertificate': True
                }
                
                # Check for a cookies file to handle YouTube authentication / bot protection
                if os.path.exists("cookies.txt"):
                    ydl_opts['cookiefile'] = 'cookies.txt'
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(youtube_url, download=False)
                    video_url = info_dict.get("url", None)
                
                if video_url:
                    cap = cv2.VideoCapture(video_url)

                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps == 0 or np.isnan(fps): 
                        fps = 30
                    frame_duration = 1.0 / fps

                    st_count = st.empty()  
                    stframe = st.empty() 
                    
                    st.write(f"🎞️ Procesando stream a {int(fps)} FPS (Presiona 'Stop' arriba a la derecha para detener)")
                    
                    unique_eggs = set()
                    
                    while cap.isOpened():
                        start_time = time.time()
                        
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                 
                        results = model.track(
                            frame, 
                            persist=True, 
                            verbose=False, 
                            conf=conf_threshold_video,
                            iou=iou_threshold,
                            imgsz=imgsz_inference
                        )

                        filtered = filter_egg_detections(
                            results[0],
                            frame.shape,
                            conf_threshold_video,
                            min_area_percent / 100.0,
                            max_area_percent / 100.0,
                            min_aspect_ratio,
                            max_aspect_ratio,
                            use_geometry_filter and use_belt_roi,
                            min(roi_y_min, roi_y_max),
                            max(roi_y_min, roi_y_max),
                        )
                        
                
                        tracked_ids = [d["id"] for d in filtered if d["id"] is not None]
                        if tracked_ids:
                            unique_eggs.update(tracked_ids)
                            
               
                        st_count.markdown(f"### 🥚 Total de huevos contados: **{len(unique_eggs)}**")
                        
         
                        class_names = results[0].names if hasattr(results[0], "names") else {0: "huevo"}
                        res_plotted = draw_filtered_detections(frame, filtered, class_names)
                        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                        
           
                        stframe.image(res_plotted_rgb, channels="RGB")
                        
             
                        elapsed_time = time.time() - start_time
                        sleep_time = frame_duration - elapsed_time
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                    cap.release()
            except ImportError:
                st.error("Por favor instala yt-dlp usando: pip install yt-dlp")
            except Exception as e:
                st.error(f"Ocurrió un error al procesar el video: {e}")

