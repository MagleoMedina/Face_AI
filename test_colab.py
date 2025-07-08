import sys
import os
import cv2
import numpy as np
from keras.models import load_model

# Constantes
IMG_SIZE = (224, 224)
EMOTION_CLASSES = ['alegre','cansado','ira','pensativo','riendo','sorprendido','triste']
PERSON_CLASSES = ['Magleo', 'Hector']
MODEL_PATH = 'model_emotions.keras'

def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {img_path}")
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(model, img_array):
    preds = model.predict(img_array)
    emotion_pred = np.argmax(preds[0], axis=1)[0]
    person_pred = np.argmax(preds[1], axis=1)[0]
    return EMOTION_CLASSES[emotion_pred], PERSON_CLASSES[person_pred]

if __name__ == "__main__":
    # Asigna aquí el nombre del archivo de imagen que deseas predecir
    nombre_archivo = "feli.jpg"  # Cambia este valor por el nombre de tu imagen

    img_path = os.path.join(os.getcwd(), nombre_archivo)

    if not os.path.exists(MODEL_PATH):
        print(f"Modelo no encontrado en {MODEL_PATH}")
        sys.exit(1)

    if not os.path.exists(img_path):
        print(f"Imagen no encontrada: {img_path}")
        sys.exit(1)

    try:
        img_array = load_and_preprocess_image(img_path)
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        sys.exit(1)

    model = load_model(MODEL_PATH)
    emotion, person = predict_image(model, img_array)

    print(f"Archivo: {nombre_archivo}")
    print(f"Predicción de emoción: {emotion}")
    print(f"Predicción de persona: {person}")
