# test.py

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf

# --- Constantes y Configuraciones ---
MODEL_PATH = 'model_emotions.keras'
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.7 # Umbral de confianza para determinar si es "DESCONOCIDO"

# Clases (deben coincidir con el script de entrenamiento)
EMOTION_CLASSES = ['alegre','cansado','ira','pensativo','riendo','sorprendido','triste']
PERSON_CLASSES = ['Magleo', 'Hector']


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Detector de Emociones y Personas - UNEG")
        self.geometry("600x550")
        self.resizable(False, False)
        ctk.set_appearance_mode("dark")

        # Cargar el modelo
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Modelo cargado exitosamente.")
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            self.model = None

        # Widgets de la Interfaz 
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.load_button = ctk.CTkButton(self.main_frame, text="Cargar Imagen", command=self.load_and_predict_image)
        self.load_button.pack(pady=10)
        
        # Etiqueta para mostrar la imagen
        self.image_label = ctk.CTkLabel(self.main_frame, text="")
        self.image_label.pack(pady=10)
        
        # Frame para los resultados
        self.results_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.results_frame.pack(pady=20, padx=20)

        self.person_label = ctk.CTkLabel(self.results_frame, text="Persona: --", font=("Helvetica", 18, "bold"))
        self.person_label.grid(row=0, column=0, padx=20)

        self.emotion_label = ctk.CTkLabel(self.results_frame, text="Emoción: --", font=("Helvetica", 18, "bold"))
        self.emotion_label.grid(row=0, column=1, padx=20)


    def load_and_predict_image(self):
        """
        Abre un diálogo para seleccionar una imagen, la procesa y realiza la predicción.
        """
        if not self.model:
            self.person_label.configure(text="Error: Modelo no cargado")
            return

        file_path = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )

        if not file_path:
            return

        # Cargar y mostrar la imagen en la GUI
        pil_image = Image.open(file_path)
        display_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(300, 300))
        self.image_label.configure(image=display_image)
        
        # Preprocesar la imagen para el modelo
        try:
            img_array = cv2.imread(file_path)
            img_resized = cv2.resize(img_array, IMG_SIZE)
            img_normalized = img_resized / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
        except Exception as e:
            self.person_label.configure(text=f"Error al procesar imagen: {e}")
            return
            
        # Realizar la predicción
        predictions = self.model.predict(img_batch)
        emotion_preds = predictions[0][0]
        person_preds = predictions[1][0]

        # Interpretar los resultados
        # Emoción
        emotion_index = np.argmax(emotion_preds)
        predicted_emotion = EMOTION_CLASSES[emotion_index]

        # Persona
        person_index = np.argmax(person_preds)
        person_confidence = person_preds[person_index]

        if person_confidence >= CONFIDENCE_THRESHOLD:
            predicted_person = PERSON_CLASSES[person_index]
        else:
            predicted_person = "DESCONOCIDO"

        # Actualizar las etiquetas de resultados
        self.person_label.configure(text=f"Persona: {predicted_person}")
        self.emotion_label.configure(text=f"Emoción: {predicted_emotion}")


if __name__ == "__main__":
    app = App()
    app.mainloop()