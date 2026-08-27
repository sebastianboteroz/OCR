import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

# ---------- Configuración de página ----------
st.set_page_config(
    page_title="OCR Vision - Reconocimiento de Texto",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Estilos personalizados (Dark + Blue Gradient + Responsive) ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    /* Variables de diseño */
    :root {
        --bg-dark: #080B10;
        --card-bg: rgba(16, 22, 34, 0.65);
        --card-border: rgba(56, 189, 248, 0.18);
        --accent-blue: #38BDF8;
        --accent-glow: #1E40AF;
        --text-primary: #F3F4F6;
        --text-secondary: #9CA3AF;
    }

    /* Fondo global con gradiente azul profundo */
    .stApp {
        background: radial-gradient(circle at 50% -20%, #1E3A8A 0%, #0F172A 45%, #030712 100%) !important;
        background-attachment: fixed !important;
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar adaptada al tema oscuro */
    section[data-testid="stSidebar"] {
        background-color: rgba(3, 7, 18, 0.85) !important;
        border-right: 1px solid var(--card-border) !important;
        backdrop-filter: blur(12px);
    }

    /* Contenedor principal responsive */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Encabezado Principal */
    .eyebrow {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.85rem;
        letter-spacing: 0.2em;
        color: var(--accent-blue);
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 600;
    }
    .eyebrow::before {
        content: "";
        width: 8px;
        height: 8px;
        background: var(--accent-blue);
        border-radius: 50%;
        box-shadow: 0 0 10px var(--accent-blue);
        display: inline-block;
    }

    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.2rem, 5vw, 4rem);
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin: 0 0 0.8rem 0;
        background: linear-gradient(135deg, #FFFFFF 30%, var(--accent-blue) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        font-size: clamp(0.95rem, 2vw, 1.15rem);
        color: var(--text-secondary);
        margin-bottom: 2rem;
        max-width: 60ch;
        font-weight: 300;
    }

    hr.divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, var(--card-border) 0%, rgba(255,255,255,0) 100%);
        margin: 1.5rem 0 2.5rem 0;
    }

    /* Contenedores de paneles */
    .panel-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.8rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--accent-blue);
        border-bottom: 1px solid var(--card-border);
        padding-bottom: 0.5rem;
        margin-bottom: 1.2rem;
        font-weight: 600;
    }

    /* Visor de la cámara personalizado */
    div[data-testid="stCameraInput"] {
        border: 1px solid var(--card-border);
        border-radius: 12px;
        background: var(--card-bg);
        backdrop-filter: blur(8px);
        padding: 10px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Caja de resultado de texto */
    .result-box {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-left: 4px solid var(--accent-blue);
        border-radius: 8px;
        padding: 1.4rem;
        font-family: 'Inter', monospace;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #E5E7EB;
        white-space: pre-wrap;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
        min-height: 150px;
    }

    /* Estilización de Botones */
    .stButton>button, .stDownloadButton>button {
        border-radius: 8px !important;
        border: 1px solid var(--accent-blue) !important;
        background: linear-gradient(135deg, #1E3A8A 0%, #0284C7 100%) !important;
        color: #FFFFFF !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.2) !important;
        width: 100%;
    }

    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 25px rgba(56, 189, 248, 0.5) !important;
        border-color: #38BDF8 !important;
    }

    /* Ajustes responsivos para pantallas pequeñas */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
        }
        .main-title {
            text-align: left;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Encabezado ----------
st.markdown('<p class="eyebrow">Instrumento de Lectura Óptica</p>', unsafe_allow_html=True)
st.markdown('<p class="main-title">OCR_Vision</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Captura imágenes mediante la cámara o carga un archivo para extraer texto de forma precisa en tiempo real.</p>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ---------- Sidebar: Opciones ----------
with st.sidebar:
    st.header("⚙️ Configuración")

    with st.expander("Filtros de Imagen", expanded=True):
        filtro = st.radio(
            "Preprocesamiento",
            ('Sin Filtro', 'Escala de grises', 'Blanco y negro (umbral)', 'Invertir colores'),
            help="El preprocesamiento ayuda a reducir el ruido visual y mejorar la detección de caracteres."
        )
        umbral = 128
        if filtro == 'Blanco y negro (umbral)':
            umbral = st.slider("Sensibilidad del umbral", 0, 255, 128)

    with st.expander("Idioma de Reconocimiento", expanded=True):
        idioma = st.selectbox("Idioma Tesseract", ("Español", "Inglés"), index=0)
        lang_code = "spa" if idioma == "Español" else "eng"

# ---------- Selector de fuente de entrada ----------
opcion_entrada = st.radio(
    "Selecciona el método de entrada:",
    ("📷 Cámara en vivo", "📁 Cargar archivo de imagen"),
    horizontal=True
)

img_cv = None

if opcion_entrada == "📷 Cámara en vivo":
    img_file_buffer = st.camera_input("Capturar imagen")
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        img_cv = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

else:
    uploaded_file = st.file_uploader("Sube una imagen (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# ---------- Procesamiento y renderizado ----------
if img_cv is not None:
    # Aplicación de filtros OpenCV
    if filtro == 'Escala de grises':
        procesada = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    elif filtro == 'Blanco y negro (umbral)':
        gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, procesada = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY)
    elif filtro == 'Invertir colores':
        procesada = cv2.bitwise_not(img_cv)
    else:
        procesada = img_cv

    # Conversión para despliegue en Streamlit
    if len(procesada.shape) == 2:
        img_rgb = procesada
    else:
        img_rgb = cv2.cvtColor(procesada, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown('<p class="panel-label">01 · Vista Previa de Imagen</p>', unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True)

    with col2:
        st.markdown('<p class="panel-label">02 · Resultado del OCR</p>', unsafe_allow_html=True)
        try:
            with st.spinner("Procesando y analizando texto..."):
                text = pytesseract.image_to_string(img_rgb, lang=lang_code)

            if text.strip():
                st.markdown(f'<div class="result-box">{text}</div>', unsafe_allow_html=True)

                palabras = len(text.split())
                caracteres = len(text)
                st.caption(f"📊 Métricas: {palabras} palabras | {caracteres} caracteres")

                st.download_button(
                    label=" Descargar texto (.txt)",
                    data=text,
                    file_name="texto_extraido.txt",
                    mime="text/plain"
                )
            else:
                st.warning("No se detectó texto legible. Ajusta los filtros o intenta con una imagen con mayor contraste.")

        except pytesseract.TesseractNotFoundError:
            st.error("⚠️ Tesseract OCR no está instalado en el servidor/sistema. Asegúrate de incluir las dependencias requeridas.")
        except Exception as e:
            st.error(f"Error durante el procesamiento: {e}")

else:
    st.info("Utiliza la cámara o sube un archivo para iniciar la lectura del texto.")
