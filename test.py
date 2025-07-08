# test.py

import os
import json
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2 # Importamos OpenCV

# --- Configuración ---
MODEL_FILENAME = 'emotion_model.keras'
CLASSES_FILENAME = 'class_names.json'
PERSONS_FILENAME = 'person_names.json'
IMG_HEIGHT = 160
IMG_WIDTH = 160

# --- Cargar Modelo, Clases y Nombres de Personas ---
try:
    model = tf.keras.models.load_model(MODEL_FILENAME)
    with open(CLASSES_FILENAME, 'r') as f:
        class_names = json.load(f)
    # Intentar cargar los nombres de personas (opcional)
    if os.path.exists(PERSONS_FILENAME):
        with open(PERSONS_FILENAME, 'r') as f:
            person_map = json.load(f)
    else:
        person_map = None
    model_loaded = True
    print("✅ Modelo, clases y nombres de personas cargados correctamente.")
except Exception as e:
    model = None
    class_names = None
    person_map = None
    model_loaded = False
    print(f"❌ Error al cargar el modelo, las clases o los nombres de personas: {e}")
    print(f"Asegúrate de que los archivos '{MODEL_FILENAME}' y '{CLASSES_FILENAME}' existan.")

def get_person_name_from_filename(filename):
    """Extrae el nombre de la persona desde el nombre del archivo."""
    base = os.path.basename(filename)
    if base.startswith("Magleo_"):
        return "Magleo"
    elif base.startswith("Hector_"):
        return "Hector"
    else:
        return "desconocido"

def predict_emotion(image_path):
    """
    Carga una imagen con OpenCV, detecta el rostro, la preprocesa y predice la emoción.
    """
    if not model:
        return "Error: Modelo no cargado.", None

    img = cv2.imread(image_path)
    if img is None:
        return "Error: No se pudo cargar la imagen.", None

    # Cargar el clasificador Haar Cascade para detección de rostros
    haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    if face_cascade.empty():
        return "Error: No se pudo cargar el clasificador Haar Cascade.", None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Parámetros ajustados para ampliar el rango de detección
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # Más sensible (antes 1.1)
        minNeighbors=2     # Menos estricto (antes 4)
    )
    if len(faces) == 0:
        return "Error: No se detectó rostro en la imagen.", None

    # Tomar el primer rostro detectado
    (x, y, w, h) = faces[0]
    face_img = img[y:y+h, x:x+w]
    resized_img = cv2.resize(face_img, (IMG_WIDTH, IMG_HEIGHT))
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    img_array = np.array(rgb_img)
    img_array = np.expand_dims(img_array, 0)

    # Realizar la predicción
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_emotion = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    # --- Nuevo: obtener nombre de persona ---
    person_name = get_person_name_from_filename(image_path)
    result_text = f"Persona: {person_name} | Emoción: {predicted_emotion} ({confidence:.2f}%)"
    
    print(f"Imagen: {os.path.basename(image_path)} -> {result_text}")
    return result_text, rgb_img # Devolvemos la imagen procesada (RGB)

def select_image():
    """Abre un diálogo para seleccionar una imagen y muestra la predicción."""
    if not model_loaded:
        messagebox.showerror("Error", f"No se pudo cargar el modelo desde '{MODEL_FILENAME}'.")
        return

    file_path = filedialog.askopenfilename(
        title="Selecciona una imagen de un rostro",
        filetypes=[("Archivos de Imagen", "*.png *.jpg *.jpeg *.bmp")]
    )
    if not file_path:
        return

    prediction_text, processed_img = predict_emotion(file_path)

    if processed_img is not None:
        # Convertir la imagen procesada de OpenCV/Numpy a un formato para Tkinter
        img_pil = Image.fromarray(processed_img)
        img_pil.thumbnail((350, 350))
        img_tk = ImageTk.PhotoImage(img_pil)
        
        image_label.config(image=img_tk)
        image_label.image = img_tk
        result_label.config(text=prediction_text, fg="blue")
    else:
        messagebox.showerror("Error de Imagen", prediction_text)

def create_gui():
    """Crea y ejecuta la interfaz gráfica con Tkinter."""
    root = tk.Tk()
    root.title("Clasificador de Emociones Faciales")
    root.geometry("450x550")
    root.configure(bg="#f0f0f0")
    
    main_frame = Frame(root, bg="#f0f0f0", padx=20, pady=20)
    main_frame.pack(expand=True, fill=tk.BOTH)

    Label(main_frame, text="Probador de Modelo de Emociones", font=("Helvetica", 16, "bold"), bg="#f0f0f0").pack(pady=(0, 20))
    
    global image_label
    image_label = Label(main_frame, bg="#e0e0e0", text="Selecciona una imagen para verla aquí", width=50, height=20)
    image_label.pack(pady=10)

    global result_label
    result_label = Label(main_frame, text="La predicción aparecerá aquí", font=("Helvetica", 14), bg="#f0f0f0", fg="black")
    result_label.pack(pady=10)

    Button(main_frame, text="🖼️ Seleccionar Imagen y Predecir", command=select_image, font=("Helvetica", 12), bg="#007BFF", fg="white", relief=tk.RAISED, borderwidth=2).pack(pady=20, ipady=5, ipadx=10)
    
    root.mainloop()

if __name__ == "__main__":
    create_gui()