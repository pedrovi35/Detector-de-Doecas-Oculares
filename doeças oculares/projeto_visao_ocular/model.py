import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def build_model(input_shape, num_classes):
    """
    Constrói um modelo de CNN usando Transfer Learning com MobileNetV2.

    Args:
        input_shape (tuple): A forma da imagem de entrada (altura, largura, canais).
        num_classes (int): O número de classes de saída.

    Returns:
        tensorflow.keras.Model: O modelo compilado.
    """
    # Modelo base pré-treinado na ImageNet
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    # Congelar as camadas do modelo base para não serem treinadas inicialmente
    base_model.trainable = False

    # Adicionar camadas personalizadas no topo
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)  # Dropout para regularização
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    # Compilar o modelo
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', # Usamos one-hot encoding
                  metrics=['accuracy'])
    
    model.summary()
    return model

def train_model(model, X_train, y_train, X_val, y_val, model_path, epochs=30, patience=5):
    """
    Treina o modelo com os dados fornecidos.

    Args:
        model (tensorflow.keras.Model): O modelo a ser treinado.
        X_train, y_train: Dados e rótulos de treino.
        X_val, y_val: Dados e rótulos de validação.
        model_path (str): Caminho para salvar o melhor modelo.
        epochs (int): Número máximo de épocas.
        patience (int): Paciência para o EarlyStopping.

    Returns:
        tensorflow.keras.callbacks.History: Histórico do treinamento.
    """
    # Callback para parar o treino se a perda de validação não melhorar
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    # Callback para salvar o melhor modelo durante o treinamento
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max')

    print("\nIniciando o treinamento do modelo...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
        batch_size=32
    )
    print("Treinamento concluído.")
    return history
