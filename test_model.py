# test_model.py

import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# --- Constantes y Configuraciones ---
DATA_DIR = 'images'
IMG_SIZE = (224, 224) # Tamaño estándar para las imágenes
EPOCHS = 14 # Puedes considerar aumentar esto a 30-50 si el modelo sigue aprendiendo
BATCH_SIZE = 32

# Clases de emociones y personas
EMOTION_CLASSES = ['alegre','cansado','ira','pensativo','riendo','sorprendido','triste']
PERSON_CLASSES = ['Magleo', 'Hector']

def load_data():
    """
    Carga las imágenes desde el directorio y extrae las etiquetas de emoción y persona.
    """
    images = []
    emotion_labels = []
    person_labels = []

    print(f"Cargando imágenes desde el directorio: {DATA_DIR}")

    # Mapeo de nombres a números para las etiquetas
    emotion_map = {name: i for i, name in enumerate(EMOTION_CLASSES)}
    person_map = {name: i for i, name in enumerate(PERSON_CLASSES)}

    for emotion_folder in os.listdir(DATA_DIR):
        emotion_path = os.path.join(DATA_DIR, emotion_folder)
        if not os.path.isdir(emotion_path) or emotion_folder not in EMOTION_CLASSES:
            continue

        emotion_label = emotion_map[emotion_folder]
        print(f"Procesando carpeta: {emotion_folder}")

        for img_file in os.listdir(emotion_path):
            try:
                # Determinar la persona a partir del nombre del archivo
                if img_file.lower().startswith('magleo_'):
                    person_label = person_map['Magleo']
                elif img_file.lower().startswith('hector_'):
                    person_label = person_map['Hector']
                else:
                    continue # Ignorar si no corresponde a ninguna persona

                # Cargar y preprocesar la imagen
                img_path = os.path.join(emotion_path, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, IMG_SIZE)
                img = img / 255.0  # Normalizar

                images.append(img)
                emotion_labels.append(emotion_label)
                person_labels.append(person_label)
                # print(f"Imagen procesada: {img_file} | Emoción: {emotion_folder} | Persona: {PERSON_CLASSES[person_label]}")

            except Exception as e:
                print(f"Error al procesar la imagen {img_file}: {e}")

    print(f"Carga de datos completa. Total de imágenes: {len(images)}")
    return np.array(images), np.array(emotion_labels), np.array(person_labels)

def build_model():
    """
    Construye un modelo CNN con dos ramas de salida (multi-output), sin aumento de datos.
    """
    # Entrada
    input_layer = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name='input_layer')

    # Base Convolucional (sin capas de aumento de datos aquí)
    x = Conv2D(32, (3, 3), activation='relu')(input_layer) # Conecta directamente a input_layer
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x) # Dropout general para la base convolucional
    x = Flatten()(x)

    # Rama 1: Clasificación de Emociones
    emotion_branch = Dense(256, activation='relu')(x)
    emotion_branch = Dropout(0.4)(emotion_branch)
    emotion_branch = Dense(128, activation='relu')(emotion_branch)
    emotion_output = Dense(len(EMOTION_CLASSES), activation='softmax', name='emotion_output')(emotion_branch)

    # Rama 2: Clasificación de Personas
    person_branch = Dense(64, activation='relu')(x)
    person_branch = Dropout(0.5)(person_branch)
    person_output = Dense(len(PERSON_CLASSES), activation='softmax', name='person_output')(person_branch)

    # Crear y compilar el modelo
    model = Model(inputs=input_layer, outputs=[emotion_output, person_output], name="emotion_person_classifier")

    model.compile(
        optimizer='adam',
        loss={
            'emotion_output': 'categorical_crossentropy',
            'person_output': 'categorical_crossentropy'
        },
        loss_weights={
            'emotion_output': 2, # Mayor peso para la pérdida de emoción
            'person_output': 0.5   # Menor peso para la pérdida de persona
        },
        metrics={
            'emotion_output': 'accuracy',
            'person_output': 'accuracy'
        }
    )
    return model

if __name__ == "__main__":
    # Cargar y preparar los datos
    images, emotion_labels, person_labels = load_data()

    if len(images) == 0:
        print("No se encontraron imágenes. Asegúrate de que la estructura de carpetas sea correcta.")
    else:
        # Convertir etiquetas a formato categórico (one-hot encoding)
        y_emotion = to_categorical(emotion_labels, num_classes=len(EMOTION_CLASSES))
        y_person = to_categorical(person_labels, num_classes=len(PERSON_CLASSES))

        # Dividir los datos en conjuntos de entrenamiento y validación
        X_train, X_val, y_emotion_train, y_emotion_val, y_person_train, y_person_val = train_test_split(
            images, y_emotion, y_person, test_size=0.3, random_state=42, stratify=emotion_labels
        )

        # Construir el modelo
        model = build_model()
        model.summary()

        # Entrenar el modelo
        print("\n--- Iniciando Entrenamiento ---")
        history = model.fit(
            X_train,
            {'emotion_output': y_emotion_train, 'person_output': y_person_train},
            validation_data=(X_val, {'emotion_output': y_emotion_val, 'person_output': y_person_val}),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )

        # Guardar el modelo entrenado
        model.save('model_emotions.keras')
        print("\n✅ Entrenamiento finalizado. Modelo guardado como 'model_emotions.keras'")

        # --- Evaluación Adicional para Emociones (Matriz de Confusión) ---
        print("\n--- Evaluación de la Clasificación de Emociones ---")
        emotion_val_predictions, _ = model.predict(X_val)
        y_pred_emotion = np.argmax(emotion_val_predictions, axis=1)
        y_true_emotion = np.argmax(y_emotion_val, axis=1)

        cm = confusion_matrix(y_true_emotion, y_pred_emotion)
        print("\nMatriz de Confusión para Emociones:")
        print(cm)

        report = classification_report(y_true_emotion, y_pred_emotion, target_names=EMOTION_CLASSES)
        print("\nReporte de Clasificación para Emociones:")
        print(report)