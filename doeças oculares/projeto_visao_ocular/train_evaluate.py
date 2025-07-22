import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from preprocessing import load_and_preprocess_data
from model import build_model, train_model

# --- Configurações ---
DATASET_PATH = 'data/'
IMG_SIZE = (224, 224)
MODEL_SAVE_PATH = 'saved_model/eye_disease_model.h5'

if __name__ == '__main__':
    # 1. Carregar e pré-processar os dados
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = load_and_preprocess_data(DATASET_PATH, img_size=IMG_SIZE)
    
    num_classes = len(class_names)
    input_shape = (*IMG_SIZE, 3)

    # 2. Construir o modelo
    model = build_model(input_shape, num_classes)

    # 3. Treinar o modelo
    history = train_model(model, X_train, y_train, X_val, y_val, model_path=MODEL_SAVE_PATH)
    
    # 4. Carregar o melhor modelo salvo pelo ModelCheckpoint
    print("\nCarregando o melhor modelo salvo para avaliação...")
    best_model = load_model(MODEL_SAVE_PATH)

    # 5. Avaliar o modelo no conjunto de teste
    print("\nAvaliando o modelo no conjunto de teste...")
    loss, accuracy = best_model.evaluate(X_test, y_test)
    print(f"Acurácia no teste: {accuracy:.4f}")
    print(f"Perda no teste: {loss:.4f}")

    # 6. Gerar relatório de classificação e matriz de confusão
    y_pred_probs = best_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nRelatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Plotar a matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão')
    plt.xlabel('Previsão')
    plt.ylabel('Verdadeiro')
    plt.savefig('confusion_matrix.png')
    print("Matriz de confusão salva como 'confusion_matrix.png'")
    # plt.show() # Descomente se quiser exibir o gráfico
