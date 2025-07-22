# =========================================================================================
# PROJETO COMPLETO DE DETECÇÃO DE DOENÇAS OCULARES PARA GOOGLE COLAB
# Autor: Gemini (Google AI)
# Descrição: Um script Python que treina um modelo de Deep Learning e realiza
#            previsões de doenças oculares (Catarata, Glaucoma, Degeneração Macular, Normal)
#            a partir de imagens da retina. Inclui simulações visuais e Grad-CAM.
#            Adaptado para execução em notebooks (sem Streamlit).
# =========================================================================================

# --- 1. IMPORTAÇÕES ---
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import random
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from IPython.display import display

# --- 2. CONFIGURAÇÕES GLOBAIS E CONSTANTES ---
# Descompactar o dataset (assumindo que 'archive.zip' foi enviado para o Colab)
if os.path.exists('archive.zip'):
    print("Descompactando o dataset...")
    with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
        zip_ref.extractall('dataset')
    DATASET_PATH = 'dataset/Ocular Disease Recognition'
else:
    # Se você já descompactou manualmente e colocou na pasta 'data'
    DATASET_PATH = 'data/'

MODEL_SAVE_DIR = 'saved_model/'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'eye_disease_model.h5')

# Parâmetros
IMG_SIZE = (224, 224)
INPUT_SHAPE = (*IMG_SIZE, 3)
CLASS_NAMES = ['cataract', 'glaucoma', 'macular_degeneration', 'normal']
NUM_CLASSES = len(CLASS_NAMES)

# Criar diretório para salvar o modelo, se não existir
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

# --- 3. FUNÇÕES DE PRÉ-PROCESSAMENTO, MODELO E TREINAMENTO ---

def load_and_preprocess_data(dataset_path, img_size=(224, 224), test_split=0.15, val_split=0.15):
    """
    Carrega, pré-processa e divide as imagens para o treinamento.
    """
    print("Iniciando carregamento e pré-processamento dos dados...")
    images = []
    labels = []

    # Ajusta os nomes das classes para corresponderem aos diretórios do dataset
    class_names_from_dir = sorted(os.listdir(dataset_path))
    class_map = {name: i for i, name in enumerate(class_names_from_dir)}

    for class_name in class_names_from_dir:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        
        print(f"Carregando classe: {class_name}")
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(class_map[class_name])
            except Exception as e:
                print(f"  Erro ao carregar a imagem {img_path}: {e}")

    print("\nNormalizando e dividindo os dados...")
    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels)
    labels_one_hot = to_categorical(labels, num_classes=len(class_names_from_dir))

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, labels_one_hot, test_size=test_split, random_state=42, stratify=labels)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_split / (1 - test_split), random_state=42, stratify=y_train_val.argmax(axis=1))
    
    print("\nDados carregados com sucesso!")
    print(f"  - Conjunto de Treino: {len(X_train)} imagens")
    print(f"  - Conjunto de Validação: {len(X_val)} imagens")
    print(f"  - Conjunto de Teste: {len(X_test)} imagens")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names_from_dir

def build_model(input_shape, num_classes):
    """Constrói um modelo de CNN usando Transfer Learning com MobileNetV2."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_training_history(history):
    """Plota as curvas de acurácia e perda do treinamento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de Acurácia
    ax1.plot(history.history['accuracy'], label='Acurácia de Treino')
    ax1.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    ax1.set_title('Acurácia do Modelo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Acurácia')
    ax1.legend(loc='lower right')

    # Gráfico de Perda
    ax2.plot(history.history['loss'], label='Perda de Treino')
    ax2.plot(history.history['val_loss'], label='Perda de Validação')
    ax2.set_title('Perda do Modelo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Perda')
    ax2.legend(loc='upper right')

    plt.show()

# --- 4. FUNÇÕES DE VISUALIZAÇÃO E SIMULAÇÃO ---

# (As funções de simulação são as mesmas)
def simulate_cataract(image):
    return cv2.GaussianBlur(image, (31, 31), 0)

def simulate_glaucoma(image):
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w//2, h//2), min(h,w)//4, 255, -1)
    return cv2.bitwise_and(image, image, mask=mask)

def simulate_macular_degeneration(image):
    h, w, _ = image.shape
    sim_img = image.copy()
    cv2.circle(sim_img, (w//2, h//2), min(h,w)//6, (0, 0, 0), -1)
    return sim_img
    
# (As funções de Grad-CAM são as mesmas)
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam_on_image(img, heatmap, alpha=0.5):
    if isinstance(img, Image.Image): img = np.array(img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    return np.clip(superimposed_img, 0, 255).astype(np.uint8)

def predict_and_visualize(model, image_array, original_image, class_names_map):
    """Faz a predição e exibe todas as visualizações para uma imagem."""
    # Prepara a imagem para predição
    pred_image_array = np.expand_dims(image_array, axis=0)

    # Predição
    prediction = model.predict(pred_image_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names_map[predicted_class_index]
    confidence = np.max(prediction)
    
    print(f"\n--- Resultado da Predição ---")
    print(f"Classe Prevista: {predicted_class_name.title()} (Confiança: {confidence:.2%})")

    # Grad-CAM
    last_conv_layer_name = [layer.name for layer in model.layers if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))][-1]
    heatmap = make_gradcam_heatmap(pred_image_array, model, last_conv_layer_name)
    gradcam_img = overlay_gradcam_on_image(original_image, heatmap)

    # Simulação
    simulated_img = original_image.copy()
    if predicted_class_name == 'cataract':
        simulated_img = simulate_cataract(simulated_img)
    elif predicted_class_name == 'glaucoma':
        simulated_img = simulate_glaucoma(simulated_img)
    elif predicted_class_name == 'macular_degeneration':
        simulated_img = simulate_macular_degeneration(simulated_img)

    # Plotar resultados
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].imshow(original_image)
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')

    axes[1].imshow(gradcam_img)
    axes[1].set_title('Explicação (Grad-CAM)')
    axes[1].axis('off')

    axes[2].imshow(simulated_img)
    axes[2].set_title(f'Simulação de Visão ({predicted_class_name.title()})')
    axes[2].axis('off')
    
    plt.suptitle(f"Análise da Imagem: {predicted_class_name.title()}", fontsize=16)
    plt.show()


# --- 5. FLUXO DE EXECUÇÃO PRINCIPAL ---

if __name__ == '__main__':
    # ETAPA 1: Carregar os dados
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names_from_dir = load_and_preprocess_data(DATASET_PATH, img_size=IMG_SIZE)
    
    # ETAPA 2: Construir e compilar o modelo
    print("\nConstruindo o modelo...")
    model = build_model(INPUT_SHAPE, len(class_names_from_dir))
    model.summary()
    
    # ETAPA 3: Treinar o modelo
    print("\nIniciando o treinamento do modelo...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

    history = model.fit(
        X_train, y_train,
        epochs=25,  # Reduzido para um treinamento mais rápido no Colab
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
        batch_size=32
    )

    # ETAPA 4: Visualizar o histórico de treinamento
    print("\nHistórico de Treinamento:")
    plot_training_history(history)
    
    # ETAPA 5: Avaliar o modelo no conjunto de teste
    print("\n--- Avaliação Final no Conjunto de Teste ---")
    best_model = load_model(MODEL_SAVE_PATH)
    loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"Acurácia no Teste: {accuracy:.2%}")
    print(f"Perda no Teste: {loss:.4f}")

    # Relatório de Classificação e Matriz de Confusão
    y_pred_probs = best_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nRelatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=class_names_from_dir))

    print("Matriz de Confusão:")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_from_dir, yticklabels=class_names_from_dir)
    plt.title('Matriz de Confusão')
    plt.xlabel('Previsão')
    plt.ylabel('Verdadeiro')
    plt.show()
    
    # ETAPA 6: Realizar predição e visualização em uma imagem de teste aleatória
    print("\n--- Testando em uma Imagem Aleatória do Conjunto de Teste ---")
    # Selecionar um índice aleatório do conjunto de teste
    random_idx = random.randint(0, len(X_test) - 1)
    
    # Obter a imagem normalizada e a imagem original
    test_image_normalized = X_test[random_idx]
    test_image_original = (test_image_normalized * 255).astype(np.uint8) # Desnormalizar para visualização
    true_label_index = np.argmax(y_test[random_idx])
    true_label_name = class_names_from_dir[true_label_index]
    
    print(f"Imagem de teste selecionada. Rótulo verdadeiro: {true_label_name.title()}")
    
    # Fazer a predição e visualizar
    predict_and_visualize(best_model, test_image_normalized, test_image_original, class_names_from_dir)
