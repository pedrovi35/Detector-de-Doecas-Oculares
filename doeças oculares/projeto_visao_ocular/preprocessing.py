import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(dataset_path, img_size=(224, 224), test_split=0.15, val_split=0.15):
    """
    Carrega as imagens do dataset, pré-processa e as divide em conjuntos de treino, validação e teste.

    Args:
        dataset_path (str): Caminho para a pasta principal do dataset.
        img_size (tuple): Tamanho para redimensionar as imagens.
        test_split (float): Proporção do dataset a ser usada para teste.
        val_split (float): Proporção do dataset de treino a ser usada para validação.

    Returns:
        tuple: Contém os conjuntos de treino, validação, teste e os nomes das classes.
               (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names
    """
    print("Iniciando carregamento e pré-processamento dos dados...")
    images = []
    labels = []

    # Obter os nomes das classes (subpastas) e ordená-los para consistência
    class_names = sorted(os.listdir(dataset_path))
    class_map = {name: i for i, name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                # Carregar imagem em cores
                img = cv2.imread(img_path)
                # Converter de BGR (padrão do OpenCV) para RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Redimensionar a imagem
                img = cv2.resize(img, img_size)
                
                images.append(img)
                labels.append(class_map[class_name])
            except Exception as e:
                print(f"Erro ao carregar a imagem {img_path}: {e}")

    # Normalizar as imagens (valores de pixel entre 0 e 1)
    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels)

    # Converter rótulos para o formato one-hot encoding
    # Usaremos 'categorical_crossentropy' como loss, então o one-hot é necessário.
    num_classes = len(class_names)
    labels_one_hot = to_categorical(labels, num_classes=num_classes)

    # Primeira divisão: separar o conjunto de teste
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, labels_one_hot, test_size=test_split, random_state=42, stratify=labels
    )

    # Segunda divisão: separar o conjunto de treino e validação a partir do restante
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_split / (1 - test_split), random_state=42, stratify=y_train_val.argmax(axis=1)
    )
    
    print(f"Dataset carregado com sucesso.")
    print(f"Formato dos dados de treino: {X_train.shape}")
    print(f"Formato dos rótulos de treino: {y_train.shape}")
    print(f"Formato dos dados de validação: {X_val.shape}")
    print(f"Formato dos dados de teste: {X_test.shape}")
    print(f"Classes encontradas: {class_names}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names
