import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import threading
import json
from ollama_client import OllamaClient
import os
import itertools
import time

# --- Constants and Configurations ---
MODEL_PATH = 'model_emotions.keras'
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.7  # Confidence threshold to determine if "UNKNOWN"
CHAT_DB_FILE = 'chat_history.json'

# Classes (must match the training script)
EMOTION_CLASSES = ['alegre','cansado','ira','pensativo','riendo','sorprendido','triste']
PERSON_CLASSES = ['Magleo', 'Hector']

def build_dynamic_prompt(person, emotion):
    if person == "DESCONOCIDO":
        system_message = (
            f"Eres un asistente de seguridad educado pero cauteloso. "
            f"Tu nombre es 'Visionary'. Estás hablando con una persona desconocida que parece {emotion}. "
            f"Debes informar que solo puedes conversar con usuarios registrados."
        )
        welcome_message = (
            f"Hola. No te reconozco y pareces {emotion}. Mis funcionalidades están reservadas para usuarios registrados."
        )
    elif person == "Magleo":
        system_message = (
            f"Eres un asistente amigable y ligeramente formal. Tu nombre es 'Visionary'. "
            f"Estás hablando con Magleo, quien parece {emotion}. Adapta tu respuesta a su estado emocional."
        )
        welcome_message = (
            f"Hola Magleo, te veo {emotion} en la foto. ¿Quieres conversar sobre cómo te sientes?"
        )
    elif person == "Hector":
        system_message = (
            f"Eres un asistente muy casual y divertido, casi como un amigo. Tu nombre es 'Visionary'. "
            f"Estás hablando con Hector, quien parece {emotion}. Usa jerga, emojis y adapta tu respuesta a su emoción."
        )
        welcome_message = (
            f"¡Hey Hector! Te veo {emotion} en la foto. ¿Qué onda? ¿Quieres platicar?"
        )
    else:
        system_message = (
            f"Eres un asistente. Tu nombre es 'Visionary'. Estás hablando con {person}, quien parece {emotion}."
        )
        welcome_message = (
            f"Hola {person}, te veo {emotion} en la foto."
        )
    return {"system_message": system_message, "welcome_message": welcome_message}

class ChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Agente de vision - UNEG")
        self.geometry("800x700")
        self.resizable(False, False)
        ctk.set_appearance_mode("dark")

        self.current_user = None
        self.current_emotion = None
        self.chat_history = []
        self.ollama = OllamaClient()
        self.chats_data = {}  # Diccionario para todos los chats
        self.current_chat_id = None
        self.chat_selector = None

        # Load the vision model
        try:
            self.vision_model = tf.keras.models.load_model(MODEL_PATH)
            print("✅Modelo keras cargado exitosamente.")
        except Exception as e:
            print(f"❌ Erros al cargar el modelo Keras {e}")
            self.vision_model = None
            messagebox.showerror("Error", "No se ha podido cargar el modelo de vision .Por favor revisa el archivo 'model_emotions.keras'.")
            self.destroy()
            return
        
        self.loading_label = None
        self.loading_thread = None
        self.loading_active = False

        self._setup_ui()
        # Mueve la carga del historial aquí, después de crear los widgets
        self.load_chat_history()

    def _setup_ui(self):
        # --- Main Frame ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.columnconfigure(0, weight=1)

        # --- Top Frame for User Identification ---
        self.top_frame = ctk.CTkFrame(self.main_frame)
        self.top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.top_frame.columnconfigure(0, weight=1)

        self.load_button = ctk.CTkButton(self.top_frame, text="Carga una imagen para identificar al usuario", command=self.identify_user_from_image)
        self.load_button.grid(row=0, column=0, pady=10, sticky="ew")

        self.image_label = ctk.CTkLabel(self.top_frame, text="")
        self.image_label.grid(row=1, column=0, pady=5, sticky="ew")
        
        self.info_label = ctk.CTkLabel(self.top_frame, text="Porfavor carga una imagen para iniciar el chat.", font=("Helvetica", 16, "bold"))
        self.info_label.grid(row=2, column=0, pady=10, sticky="ew")

        # --- Chat Frame ---
        self.chat_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.chat_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.chat_frame.rowconfigure(0, weight=1)
        self.chat_frame.columnconfigure(0, weight=1)
        self.chat_frame.rowconfigure(1, weight=0)

        self.chat_display = ctk.CTkTextbox(self.chat_frame, state="disabled", wrap="word", height=400)
        self.chat_display.grid(row=0, column=0, pady=10, padx=10, sticky="nsew")
        
        self.entry_frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        self.entry_frame.grid(row=1, column=0, pady=10, padx=10, sticky="ew")
        self.entry_frame.columnconfigure(0, weight=1)
        self.entry_frame.columnconfigure(1, weight=0)
        
        self.chat_entry = ctk.CTkEntry(self.entry_frame, placeholder_text="Escribe tu mensaje...", height=40)
        self.chat_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.chat_entry.bind("<Return>", self.send_message_event)

        self.send_button = ctk.CTkButton(self.entry_frame, text="Send", command=self.send_message)
        self.send_button.grid(row=0, column=1, sticky="ew")
        
        # --- Loading Indicator ---
        self.loading_label = ctk.CTkLabel(self.chat_frame, text="", font=("Helvetica", 14, "italic"))
        self.loading_label.grid(row=2, column=0, pady=5, sticky="ew")
        
        # --- Selector de chats ---
        self.chat_selector_frame = ctk.CTkFrame(self.main_frame)
        self.chat_selector_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0,10))
        self.chat_selector_frame.columnconfigure(0, weight=1)
        self.chat_selector = ctk.CTkComboBox(self.chat_selector_frame, values=[], command=self.on_chat_selected)
        self.chat_selector.grid(row=0, column=0, sticky="ew", padx=(0,10))
        self.new_chat_button = ctk.CTkButton(self.chat_selector_frame, text="Nuevo chat", command=self.create_new_chat)
        self.new_chat_button.grid(row=0, column=1, sticky="ew")

        # Initially, disable chat
        self.disable_chat()

    def start_loading_animation(self):
        """
        Inicia la animación de carga mostrando un mensaje dinámico.
        """
        self.loading_active = True
        self.loading_label.configure(text="Visionary está pensando en su respuesta...")
        self.loading_thread = threading.Thread(target=self._loading_animation)
        self.loading_thread.start()

    def stop_loading_animation(self):
        """
        Detiene la animación de carga.
        """
        self.loading_active = False
        if self.loading_thread:
            self.loading_thread.join()
        self.loading_label.configure(text="")

    def _loading_animation(self):
        """
        Actualiza dinámicamente el texto del indicador de carga.
        """
        dots = itertools.cycle(["", ".", "..", "..."])
        while self.loading_active:
            current_dots = next(dots)
            self.loading_label.configure(text=f"Visionary está pensando en su respuesta{current_dots}")
            time.sleep(0.5)

    def identify_user_from_image(self):
        if not self.vision_model:
            messagebox.showerror("Error", "Modelo no cargado", "No se ha podido cargar el modelo de visión. Por favor revisa el archivo 'model_emotions.keras'.")
            return

        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        # Mostrar la imagen
        pil_image = Image.open(file_path)
        # --- Corregir orientación usando EXIF ---
        try:
            exif = pil_image._getexif()
            if exif is not None:
                orientation_key = 274  # cf. ExifTags
                if orientation_key in exif:
                    orientation = exif[orientation_key]
                    if orientation == 3:
                        pil_image = pil_image.rotate(180, expand=True)
                    elif orientation == 6:
                        pil_image = pil_image.rotate(270, expand=True)
                    elif orientation == 8:
                        pil_image = pil_image.rotate(90, expand=True)
        except Exception as e:
            print(f"Error al corregir la orientación: {e}")
        display_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(150, 150))
        self.image_label.configure(image=display_image)

        # Procesar y predecir en un hilo separado para evitar congelar la interfaz
        thread = threading.Thread(target=self.predict, args=(file_path,))
        thread.start()

    def predict(self, file_path):
        try:
            # Preprocesar la imagen
            pil_image = Image.open(file_path)
            # --- Corregir orientación usando EXIF ---
            try:
                exif = pil_image._getexif()
                if exif is not None:
                    orientation_key = 274  # cf. ExifTags
                    if orientation_key in exif:
                        orientation = exif[orientation_key]
                        if orientation == 3:
                            pil_image = pil_image.rotate(180, expand=True)
                        elif orientation == 6:
                            pil_image = pil_image.rotate(270, expand=True)
                        elif orientation == 8:
                            pil_image = pil_image.rotate(90, expand=True)
            except Exception as e:
                print(f"Error al corregir la orientación: {e}")
            # --- Carga compatible con nombres UTF-8 ---
            img_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) == 0:
                self.update_ui_for_user("DESCONOCIDO", "No se detectó ningún rostro en la imagen.")
                return

            x, y, w, h = faces[0]
            face_img = img_array[y:y+h, x:x+w]
            img_resized = cv2.resize(face_img, IMG_SIZE)
            img_normalized = img_resized / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)

            # Realizar la predicción
            predictions = self.vision_model.predict(img_batch)
            emotion_preds, person_preds = predictions[0][0], predictions[1][0]
            
            # Interpretar resultados
            emotion_index = np.argmax(emotion_preds)
            predicted_emotion = EMOTION_CLASSES[emotion_index]

            person_index = np.argmax(person_preds)
            person_confidence = person_preds[person_index]

            if person_confidence >= CONFIDENCE_THRESHOLD:
                predicted_person = PERSON_CLASSES[person_index]
            else:
                predicted_person = "DESCONOCIDO"
            
            self.update_ui_for_user(predicted_person, predicted_emotion)

        except Exception as e:
            self.update_ui_for_user("DESCONOCIDO", f"Error al procesar la imagen: {e}")
    
    def create_new_chat(self):
        # Crear un nuevo chat con nombre único
        chat_name = f"Chat {len(self.chats_data)+1}"
        self.chats_data[chat_name] = {
            "user": None,
            "emotion": None,  # Guardar emoción
            "history": []
        }
        self.chat_selector.configure(values=list(self.chats_data.keys()))
        self.chat_selector.set(chat_name)
        self.current_chat_id = chat_name
        self.chat_history = []
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")
        self.info_label.configure(text="Porfavor carga una imagen para iniciar el chat.")

    def on_chat_selected(self, chat_name):
        # Cambia al chat seleccionado
        self.current_chat_id = chat_name
        chat_data = self.chats_data.get(chat_name, {"user": None, "emotion": None, "history": []})
        self.current_user = chat_data.get("user")
        self.current_emotion = chat_data.get("emotion")
        self.chat_history = chat_data.get("history", [])
        
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        for msg in self.chat_history:
            if msg.get("role") != "system":
                sender = "Visionary" if msg["role"] == "assistant" else "Tú"
                self.chat_display.insert("end", f"{sender}: {msg['content']}\n\n")
        self.chat_display.configure(state="disabled")
        
        # Mostrar emoción en la interfaz si existe
        if self.current_user:
            info = f"Persona: {self.current_user}"
            if self.current_emotion:
                info += f" | Emoción: {self.current_emotion}"
            self.info_label.configure(text=info)
            self.enable_chat()
        else:
            self.info_label.configure(text="Porfavor carga una imagen para iniciar el chat.")
            self.disable_chat()

    def update_ui_for_user(self, person, emotion):
        self.current_user = person
        self.current_emotion = emotion
        prompt_config = build_dynamic_prompt(person, emotion)
        
        info_text = f"Persona: {person} | Emoción: {emotion}"
        self.info_label.configure(text=info_text)

        # Limpiar chat anterior y configurar el nuevo
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        
        # Inicializar historial con el mensaje del sistema
        self.chat_history = [{"role": "system", "content": prompt_config["system_message"]}]
        
        # Mostrar el mensaje de bienvenida
        self.add_message("Visionary", prompt_config["welcome_message"])
        
        # Actualiza el chat actual en el diccionario
        if self.current_chat_id:
            self.chats_data[self.current_chat_id] = {
                "user": person,
                "emotion": emotion,  # Guardar emoción
                "history": self.chat_history.copy()
            }
        
        if person != "DESCONOCIDO":
            self.enable_chat()
        else:
            self.disable_chat()

    def send_message_event(self, event):
        self.send_message()

    def send_message(self):
        user_message = self.chat_entry.get().strip()
        if not user_message:
            return

        self.add_message("Tú", user_message)
        self.chat_entry.delete(0, "end")

        # Add user message to history for context
        self.chat_history.append({"role": "user", "content": user_message})
        if self.current_chat_id:
            self.chats_data[self.current_chat_id]["history"] = self.chat_history.copy()
        # Get AI response in a separate thread
        thread = threading.Thread(target=self.get_ai_response)
        thread.start()

    def get_ai_response(self):
        # Inicia el indicador de carga
        self.start_loading_animation()

        # Deshabilita la entrada mientras el modelo responde
        self.chat_entry.configure(state="disabled")
        self.send_button.configure(state="disabled")
        
        try:
            response = self.ollama.chat(model='gemma3', messages=self.chat_history)
            
            if response and "message" in response:
                ai_message = response['message']['content']
                
                # Agrega el mensaje del asistente al historial
                self.chat_history.append({"role": "assistant", "content": ai_message})
                self.add_message("Visionary", ai_message)
            else:
                self.add_message("System", "Error: No se pudo obtener una respuesta de Ollama. Asegúrate de que esté funcionando.")
        finally:
            # Detiene el indicador de carga
            self.stop_loading_animation()

            # Habilita la entrada nuevamente
            self.chat_entry.configure(state="normal")
            self.send_button.configure(state="normal")
            self.save_chat_history()

    def add_message(self, sender, message):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"{sender}: {message}\n\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.yview_moveto(1.0) # Auto-scroll
        # Actualiza el historial en el chat actual
        if self.current_chat_id:
            self.chats_data[self.current_chat_id]["history"] = self.chat_history.copy()

    def enable_chat(self):
        self.chat_entry.configure(state="normal")
        self.send_button.configure(state="normal")

    def disable_chat(self):
        self.chat_entry.configure(state="disabled")
        self.send_button.configure(state="disabled")

    def load_chat_history(self):
        try:
            if os.path.exists(CHAT_DB_FILE):
                with open(CHAT_DB_FILE, 'r', encoding="utf-8") as f:
                    self.chats_data = json.load(f)
                chat_names = list(self.chats_data.keys())
                # Si hay chats precargados, crea y selecciona uno nuevo al inicio
                new_chat_name = f"Chat {len(chat_names)+1}"
                self.chats_data[new_chat_name] = {
                    "user": None,
                    "history": []
                }
                all_chats = list(self.chats_data.keys())
                self.chat_selector.configure(values=all_chats)
                self.chat_selector.set(new_chat_name)
                self.current_chat_id = new_chat_name
                self.on_chat_selected(new_chat_name)
            else:
                self.create_new_chat()
        except Exception as e:
            print(f"No se pudo cargar el historial de chats: {e}")
            self.chats_data = {}
            self.create_new_chat()

    def save_chat_history(self):
        try:
            # Actualiza el historial del chat actual antes de guardar
            if self.current_chat_id:
                self.chats_data[self.current_chat_id]["user"] = self.current_user
                self.chats_data[self.current_chat_id]["emotion"] = getattr(self, "current_emotion", None)
                self.chats_data[self.current_chat_id]["history"] = self.chat_history.copy()
            with open(CHAT_DB_FILE, 'w', encoding="utf-8") as f:
                json.dump(self.chats_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error al guardar el historial de chats: {e}")


if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()