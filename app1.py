import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

# ---------- Configuración de la página ----------
st.set_page_config(
    page_title="Lector de Texto OCR",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Estilos Personalizados (Fondo Blanco Elegante + Alto Contraste) ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #FFFFFF;
        --bg-sidebar: #F8FAFC;
        --bg-card: #FFFFFF;
        --border-color: #E2E8F0;
        --accent-blue: #0284C7;
        --accent-hover: #0369A1;
        --text-primary: #0F172A;
        --text-secondary: #475569;
    }

    /* Fondo principal blanco */
    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }

    /* Barra lateral clara */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    /* Textos y etiquetas en negro/gris oscuro legible */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, div[data-testid="stMarkdownContainer"] {
        color: var(--text-primary) !important;
    }

    /* Textos secundarios */
    .hero-subtitle, .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-secondary) !important;
    }

    /* Badge superior */
    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #F0F9FF;
        border: 1px solid #BAE6FD;
        color: var(--accent-blue) !important;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.2;
        margin-bottom: 0.5rem;
        color: var(--text-primary) !important;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        max-width: 65ch;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }

    /* Títulos de sección */
    .card-header {
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--accent-blue) !important;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Estilo de botones */
    .stButton>button, .stDownloadButton>button {
        border-radius: 8px !important;
        border: 1px solid var(--accent-blue) !important;
        background: var(--accent-blue) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease-in-out !important;
        width: 100%;
    }

    .stButton>button:hover, .stDownloadButton>button:hover {
        background: var(--accent-hover) !important;
        border-color: var(--accent-hover) !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.2) !important;
    }

    /* Ajustes para cajas de texto y campos de entrada */
    textarea, input {
        border-radius: 8px !important;
        border: 1px solid var(--border-color) !important;
        background-color: #FFFFFF !important;
        color: var(--text-primary) !important;
    }

    /* Resaltado para accesibilidad */
    button:focus-visible, input:focus-visible, textarea:focus-visible {
        outline: 2px solid var(--accent-blue) !important;
        outline-offset: 2px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Encabezado Principal ----------
st.markdown('<div class="hero-tag">✨ Lector Inteligente de Texto</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Convierte tus imágenes a texto</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Toma una foto o sube una imagen desde tu dispositivo. Nosotros nos encargamos de extraer todo el texto para que puedas copiarlo o descargarlo al instante.</p>', unsafe_allow_html=True)

# ---------- Barra Lateral ----------
with st.sidebar:
    st.header("🛠️ Herramientas")
    
    st.subheader("1. Método de entrada")
    metodo = st.radio(
        "¿Cómo prefieres ingresar la imagen?",
        ("📷 Usar cámara", "📁 Subir un archivo"),
        help="Elige la opción que te sea más cómoda."
    )
    
    st.markdown("---")
    st.subheader("2. Ajuste de imagen")
    
    filtro = st.radio(
        "Mejorar lectura:",
        ('Sin cambios', 'Escala de grises', 'Blanco y negro (Umbral)', 'Invertir colores'),
        help="Si la foto tiene mala iluminación o poco contraste, probar con estos filtros ayuda al sistema a leer mejor."
    )
    
    umbral_val = 128
    if filtro == 'Blanco y negro (Umbral)':
        umbral_val = st.slider("Sensibilidad", 0, 255, 128, help="Ajusta el nivel de negro/blanco para resaltar las letras.")

    st.markdown("---")
    st.subheader("3. Idioma")
    idioma = st.selectbox("Idioma del texto:", ("Español", "Inglés"), index=0)
    lang_code = "spa" if idioma == "Español" else "eng"

# ---------- Captura / Carga de Imagen ----------
img_cv = None

if metodo == "📷 Usar cámara":
    img_buffer = st.camera_input("Toma la foto directamente desde aquí:")
    if img_buffer is not None:
        bytes_data = img_buffer.getvalue()
        img_cv = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
else:
    archivo = st.file_uploader("Selecciona una imagen (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"])
    if archivo is not None:
        image = Image.open(archivo)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# ---------- Procesamiento y Despliegue ----------
if img_cv is not None:
    # Aplicar filtros
    if filtro == 'Escala de grises':
        procesada = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    elif filtro == 'Blanco y negro (Umbral)':
        gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, procesada = cv2.threshold(gris, umbral_val, 255, cv2.THRESH_BINARY)
    elif filtro == 'Invertir colores':
        procesada = cv2.bitwise_not(img_cv)
    else:
        procesada = img_cv

    # Formato para Streamlit y OCR
    if len(procesada.shape) == 2:
        img_rgb = procesada
    else:
        img_rgb = cv2.cvtColor(procesada, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown('<div class="card-header">🖼️ 1. Imagen a procesar</div>', unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True, caption="Así es como el sistema analiza tu imagen.")

    with col2:
        st.markdown('<div class="card-header">📝 2. Texto encontrado</div>', unsafe_allow_html=True)
        
        try:
            with st.spinner("Leyendo el contenido de la imagen..."):
                texto_detectado = pytesseract.image_to_string(img_rgb, lang=lang_code)

            if texto_detectado.strip():
                st.text_area(
                    label="Texto extraído:",
                    value=texto_detectado,
                    height=240,
                    help="Puedes seleccionar y copiar directamente este texto."
                )
                
                num_palabras = len(texto_detectado.split())
                num_caracteres = len(texto_detectado)
                st.caption(f"📊 **Resumen:** {num_palabras} palabras | {num_caracteres} caracteres")
                
                st.download_button(
                    label="Descargar texto (.txt)",
                    data=texto_detectado,
                    file_name="texto_extraido.txt",
                    mime="text/plain",
                    help="Guarda el texto en un archivo de notas en tu equipo."
                )
            else:
                st.warning("🔍 No logramos encontrar texto claro en esta imagen. Intenta acercar más la cámara, mejorar la iluminación o cambiar el filtro en la barra lateral.")

        except pytesseract.TesseractNotFoundError:
            st.error("⚠️ No se encontró el motor Tesseract en el sistema. Asegúrate de tenerlo instalado en tu equipo o servidor.")
        except Exception as e:
            st.error(f"Hubo un problema inesperado al procesar la imagen: {e}")

else:
    st.info("👋 **¡Todo listo!** Toma una foto con la cámara o sube una imagen desde el menú lateral para comenzar.")
