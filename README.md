# 🥚 Detección y Clasificación de Huevos con YOLOv26

Sistema de Computer Vision interactivo construido con **YOLOv26** y desplegado con **Streamlit** para la automatización en el control de calidad de huevos. Permite la detección, clasificación (sanos vs dañados) y el conteo en tiempo real.

## ✨ Características Principales

- **Análisis de Imágenes:** Sube imágenes locales para detectar y clasificar huevos al instante.
- **Procesamiento de Video Local:** Simula un entorno industrial (como una cinta transportadora) para detectar y contar huevos frame por frame.
- **Streaming desde YouTube:** Procesa videos de YouTube en tiempo real utilizando `yt-dlp`.
- **Tracking Inteligente:** El sistema de seguimiento de YOLOv26 evita el conteo doble asignando un ID único a cada huevo mientras pasa por la cámara.

## 🛠️ Tecnologías Utilizadas

- **Modelo:** Ultralytics YOLOv26
- **Interfaz Web:** Streamlit
- **Visión por Computadora:** OpenCV / Pillow
- **Extracción de Video:** yt-dlp

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/detectar_huevos.git
cd detectar_huevos
```

### 2. Instalar las dependencias
Se recomienda utilizar un entorno virtual. Puedes instalar los requisitos utilizando instaladores rápidos como `uv` o `pip`:
```bash
pip install streamlit ultralytics opencv-python-headless pillow yt-dlp
```
*(También puedes usar tu archivo de configuración ejecutando `uv pip install --system .` o similar).*

### 3. Ejecutar la aplicación
Inicia el servidor local de Streamlit ejecutando:
```bash
streamlit run fronted/app.py
```
Abre automáticamente en tu navegador `http://localhost:8501`.

## 🍪 Nota sobre Procesamiento de Videos en YouTube

Debido a las medidas antibot de YouTube, es posible que la herramienta te solicite una verificación. Para solucionarlo y procesar videos online:

1. Instala la extensión **"Get cookies.txt LOCALLY"** en tu navegador (Chrome / Edge / Firefox).
2. Entra a YouTube (asegúrate de haber iniciado sesión) y haz clic en "Exportar" en la extensión.
3. Guarda el archivo resultante con el nombre exacto de **`cookies.txt`** en la carpeta principal de este proyecto.
4. Streamlit leerá automáticamente este archivo para autenticar la conexión.

## 📂 Estructura del Proyecto

```text
detectar_huevos/
├── fronted/             # Código de la aplicación web 
│   └── app.py           # Script principal de Streamlit
├── models/              # Pesos del modelo entrenado (ej. best.pt)
├── notebooks/           # Flujo de pruebas y entrenamiento de YOLOv26
├── pyproject.toml       # Configuración y dependencias del entorno Python
├── launch.json          # Configuraciones de depuración para VS Code
```

## 🤝 Autor
Desarrollado para la optimización de procesos de control de calidad visual.
