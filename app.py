import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

from simulation import get_simulation_for_disease

# --- Configurações da Página ---
st.set_page_config(
    page_title="Detecção de Doenças Oculares",
    page_icon="👁️",
    layout="wide"
)

# --- Funções Auxiliares ---

@st.cache_resource
def load_trained_model():
    """Carrega o modelo Keras treinado. O cache evita recarregar a cada interação."""
    try:
        model = tf.keras.models.load_model('saved_model/eye_disease_model.h5')
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        st.error("Certifique-se de que o arquivo 'eye_disease_model.h5' está na pasta 'saved_model/'.")
        st.info("Você precisa treinar o modelo primeiro executando o script 'train_evaluate.py'.")
        return None

def preprocess_image(image, img_size=(224, 224)):
    """Pré-processa a imagem para o formato esperado pelo modelo."""
    img = image.resize(img_size)
    img_array = np.array(img)
    
    # Se a imagem tiver um canal alfa (transparência), remova-o
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
        
    # Normalizar e expandir dimensões para criar um batch de 1
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Gera um heatmap Grad-CAM para uma imagem de entrada."""
    # Modelo que mapeia a imagem de entrada para as ativações da última camada conv e as predições
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Grava as operações para calcular o gradiente
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradiente da classe de saída com relação ao feature map da camada convolucional
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Média dos gradientes ao longo dos canais
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiplica cada canal no feature map pelo "peso" (gradiente)
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normaliza o heatmap entre 0 e 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam_on_image(img, heatmap, alpha=0.4):
    """Aplica o heatmap Grad-CAM sobre a imagem original."""
    # Redimensiona o heatmap para o tamanho da imagem original
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Converte a imagem original para o formato correto
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Superpõe o heatmap na imagem original
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img

# --- Carregamento do Modelo e Nomes de Classe ---
model = load_trained_model()
CLASS_NAMES = [
    'Central Serous Chorioretinopathy',
    'Diabetic Retinopathy',
    'Disc Edema',
    'Glaucoma',
    'Healthy',
    'Macular Scar',
    'Myopia',
    'Pterygium',
    'Retinal Detachment',
    'Retinitis Pigmentosa'
]

# --- Interface do Streamlit ---
st.title("👁️ Detector de Doenças Oculares")
st.markdown("Faça o upload de uma imagem de retina para classificar entre Catarata, Glaucoma, Degeneração Macular ou Normal.")

# Colunas para layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Envie sua Imagem")
    uploaded_file = st.file_uploader("Escolha uma imagem de olho...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Abrir e exibir a imagem enviada
    original_image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.image(original_image, caption="Imagem Enviada", use_column_width=True)

    # Pré-processar a imagem e fazer a predição
    processed_image = preprocess_image(original_image)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = np.max(prediction)

    with col2:
        st.header("2. Resultados da Análise")
        st.subheader("Diagnóstico Previsto:")
        st.markdown(f"**{predicted_class_name.replace('_', ' ').title()}** com **{confidence:.2%}** de confiança.")

        st.info(get_simulation_for_disease(predicted_class_name)['description'])
        
        st.subheader("Visualização da Doença (Simulação)")
        simulation_info = get_simulation_for_disease(predicted_class_name)
        simulated_vision = simulation_info['function'](np.array(original_image))
        st.image(simulated_vision, caption=f"Simulação de como seria a visão com {predicted_class_name.replace('_', ' ').title()}", use_column_width=True)

    # Explicação com Grad-CAM
    st.header("3. Entendendo a Decisão do Modelo (Grad-CAM)")
    st.markdown("O mapa de calor abaixo (Grad-CAM) destaca as regiões da imagem que mais influenciaram a decisão do modelo.")

    # Encontrar o nome da última camada convolucional automaticamente
    last_conv_layer_name = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
    
    # Gerar e exibir Grad-CAM
    heatmap = make_gradcam_heatmap(processed_image, model, last_conv_layer_name)
    gradcam_image = overlay_gradcam_on_image(original_image, heatmap)
    
    st.image(gradcam_image, caption="Explicação Visual (Grad-CAM)", use_column_width=True)

elif uploaded_file is None:
    st.info("Aguardando o upload de uma imagem.")
