import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

# ---------- Configuración de página ----------
st.set_page_config(
    page_title="OCR - Reconocimiento de Caracteres",
    page_icon="🔎",
    layout="wide"
)

# ---------- Estilos personalizados ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --paper: #FAFAF7;
        --ink: #17171A;
        --ink-soft: #6B6B65;
        --line: #DEDED4;
        --scan-red: #E8402C;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background-color: var(--paper); }

    section[data-testid="stSidebar"] {
        background-color: #F2F2EC;
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif; }

    .block-container { padding-top: 2.5rem; max-width: 1100px; }

    /* ---- Encabezado ---- */
    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.15em;
        color: var(--scan-red);
        text-transform: uppercase;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .eyebrow::before {
        content: "";
        width: 7px; height: 7px;
        background: var(--scan-red);
        border-radius: 50%;
        display: inline-block;
    }

    .main-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 4.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1;
        margin: 0 0 0.6rem 0;
        color: var(--ink);
    }
    .main-title span { color: var(--ink-soft); }

    .subtitle {
        font-size: 1.05rem;
        color: var(--ink-soft);
        margin-bottom: 2.2rem;
        max-width: 40ch;
    }

    hr.divider {
        border: none;
        border-top: 1px solid var(--line);
        margin: 1.8rem 0;
    }

    /* ---- Etiquetas tipo instrumento para cada panel ---- */
    .panel-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--ink-soft);
        border-bottom: 1px solid var(--line);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* ---- Marco tipo visor de escáner alrededor de la cámara ---- */
    div[data-testid="stCameraInput"] {
        position: relative;
        padding: 14px;
        border: 1px solid var(--line);
        background: #ffffff;
    }
    div[data-testid="stCameraInput"]::before,
    div[data-testid="stCameraInput"]::after {
        content: "";
        position: absolute;
        width: 22px; height: 22px;
        border-color: var(--scan-red);
        border-style: solid;
        z-index: 5;
    }
    div[data-testid="stCameraInput"]::before {
        top: 6px; left: 6px;
        border-width: 3px 0 0 3px;
    }
    div[data-testid="stCameraInput"]::after {
        bottom: 6px; right: 6px;
        border-width: 0 3px 3px 0;
    }

    .result-box {
        background-color: #ffffff;
        border: 1px solid var(--line);
        border-left: 3px solid var(--scan-red);
        padding: 1.2rem 1.4rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.95rem;
        line-height: 1.55;
        color: var(--ink);
        white-space: pre-wrap;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        font-family: 'IBM Plex Mono', monospace !important;
    }

    /* Botones */
    .stButton>button, .stDownloadButton>button {
        border-radius: 2px;
        border: 1px solid var(--ink);
        background: var(--ink);
        color: var(--paper);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: var(--scan-red);
        border-color: var(--scan-red);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="eyebrow">Instrumento de lectura</p>', unsafe_allow_html=True)
st.markdown('<p class="main-title">OCR<span>_</span>Vision</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Apunta, captura y extrae el texto de cualquier imagen en segundos.</p>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ---------- Sidebar: opciones ----------
with st.sidebar:
    st.header("⚙️ Configuración")

    with st.expander("Filtros de imagen", expanded=True):
        filtro = st.radio(
            "Preprocesamiento",
            ('Sin Filtro', 'Escala de grises', 'Blanco y negro (umbral)', 'Invertir colores'),
            help="La escala de grises y el umbral suelen mejorar mucho la precisión del OCR"
        )
        if filtro == 'Blanco y negro (umbral)':
            umbral = st.slider("Sensibilidad del umbral", 0, 255, 128)

    with st.expander("Idioma de reconocimiento"):
        idioma = st.selectbox("Idioma", ("Español", "Inglés"), index=0)
        lang_code = "spa" if idioma == "Español" else "eng"

# ---------- Captura de imagen ----------
img_file_buffer = st.camera_input("📷 Toma una Foto")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Aplicar filtro seleccionado
    if filtro == 'Escala de grises':
        procesada = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    elif filtro == 'Blanco y negro (umbral)':
        gris = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        _, procesada = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY)
    elif filtro == 'Invertir colores':
        procesada = cv2.bitwise_not(cv2_img)
    else:
        procesada = cv2_img

    # Preparar imagen para mostrar y para OCR
    if len(procesada.shape) == 2:  # imagen en escala de grises/binaria
        img_rgb = procesada
    else:
        img_rgb = cv2.cvtColor(procesada, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="panel-label">01 · Entrada</p>', unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True)

    with col2:
        st.markdown('<p class="panel-label">02 · Texto extraído</p>', unsafe_allow_html=True)
        try:
            with st.spinner("Leyendo texto..."):
                text = pytesseract.image_to_string(img_rgb, lang=lang_code)

            if text.strip():
                st.markdown(f'<div class="result-box">{text}</div>', unsafe_allow_html=True)

                palabras = len(text.split())
                caracteres = len(text)
                st.caption(f"📊 {palabras} palabras · {caracteres} caracteres")

                st.download_button(
                    label="⬇️ Descargar texto (.txt)",
                    data=text,
                    file_name="texto_extraido.txt",
                    mime="text/plain"
                )
            else:
                st.warning("No se detectó texto en la imagen. Prueba con otro filtro o mejor iluminación.")

        except pytesseract.TesseractNotFoundError:
            st.error("⚠️ Tesseract OCR no está instalado en este sistema. Instálalo para usar esta app.")
        except Exception as e:
            st.error(f"Ocurrió un error al procesar la imagen: {e}")

else:
    st.info("👆 Usa la cámara para capturar una imagen con texto y comenzar el reconocimiento.")
