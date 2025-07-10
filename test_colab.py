import sys
import os
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Constantes
IMG_SIZE = (224, 224)
EMOTION_CLASSES = ['alegre','cansado','ira','pensativo','riendo','sorprendido','triste']
PERSON_CLASSES = ['Magleo', 'Hector']
MODEL_PATH = 'model_emotions.keras'

def correct_image_orientation(pil_image):
    """
    Corrige la orientación de la imagen usando EXIF si está disponible.
    """
    try:
        exif = pil_image._getexif()
        if exif is not None:
            orientation_key = 274
            if orientation_key in exif:
                orientation = exif[orientation_key]
                if orientation == 3:
                    pil_image = pil_image.rotate(180, expand=True)
                elif orientation == 6:
                    pil_image = pil_image.rotate(270, expand=True)
                elif orientation == 8:
                    pil_image = pil_image.rotate(90, expand=True)
    except Exception as e:
        print(f"Advertencia al corregir orientación: {e}")
    return pil_image

def load_and_preprocess_image(img_path):
    # Cargar imagen con PIL para corregir orientación
    pil_image = Image.open(img_path)
    pil_image = correct_image_orientation(pil_image)
    img_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Detección de rostro
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        raise ValueError("No se detecta rostro en la imagen.")

    x, y, w, h = faces[0]
    face_img = img_array[y:y+h, x:x+w]
    img = cv2.resize(face_img, IMG_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(model, img_array):
    preds = model.predict(img_array)
    emotion_pred = np.argmax(preds[0], axis=1)[0]
    person_pred = np.argmax(preds[1], axis=1)[0]
    return EMOTION_CLASSES[emotion_pred], PERSON_CLASSES[person_pred]

if __name__ == "__main__":
    # Buscar todas las imágenes en la carpeta 'testing'
    testing_dir = os.path.join(os.getcwd(), "testing")
    if not os.path.exists(MODEL_PATH):
        print(f"Modelo no encontrado en {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(testing_dir):
        print(f"Carpeta 'testing' no encontrada: {testing_dir}")
        sys.exit(1)

    # Filtrar archivos de imagen comunes
    imagenes = [f for f in os.listdir(testing_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not imagenes:
        print("No se encontraron imágenes en la carpeta 'testing'.")
        sys.exit(1)

    model = load_model(MODEL_PATH)

    for nombre_archivo in imagenes:
        img_path = os.path.join(testing_dir, nombre_archivo)
        try:
            img_array = load_and_preprocess_image(img_path)
            emotion, person = predict_image(model, img_array)
            print(f"Archivo: {nombre_archivo}")
            print(f"Predicción de emoción: {emotion}")
            print(f"Predicción de persona: {person}")
            print("-" * 40)
        except Exception as e:
            print(f"Error procesando {nombre_archivo}: {e}")
            print("-" * 40)

    print("Para comparar la eficacia revisar las imagenes en la carpeta testing")
