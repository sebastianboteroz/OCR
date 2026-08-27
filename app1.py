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
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        color: #888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .result-box {
        background-color: #f7f7f9;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🔎 Reconocimiento Óptico de Caracteres</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Toma una foto y extrae el texto automáticamente</p>', unsafe_allow_html=True)

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
        st.subheader("Imagen procesada")
        st.image(img_rgb, use_container_width=True)

    with col2:
        st.subheader("Texto detectado")
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
