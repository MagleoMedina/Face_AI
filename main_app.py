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

# --- Constants and Configurations ---
MODEL_PATH = 'model_emotions.keras'
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.7  # Confidence threshold to determine if "UNKNOWN"
CHAT_DB_FILE = 'chat_history.json'

# Classes (must match the training script)
EMOTION_CLASSES = ['alegre','cansado','ira','pensativo','riendo','sorprendido','triste']
PERSON_CLASSES = ['Magleo', 'Hector']

# --- Prompts for each user ---
PROMPTS = {
    "Magleo": {
        "system_message": "You are a friendly and slightly formal assistant. Your name is 'Visionary'. You are talking to Magleo. Be helpful and always maintain a positive tone.",
        "welcome_message": "Hello Magleo, I see you in the photo. How have you been?"
    },
    "Hector": {
        "system_message": "You are a very casual and funny assistant, almost like a friend. Your name is 'Visionary'. You are talking to Hector. Use slang and emojis often.",
        "welcome_message": "Hey Hector! I see you there! What's up? 😄"
    },
    "DESCONOCIDO": {
        "system_message": "You are a polite but cautious security assistant. Your name is 'Visionary'. You are talking to an unknown person. You must inform them that you can only chat with registered users.",
        "welcome_message": "Hello. I don't recognize you. My functionalities are reserved for registered users."
    }
}

class ChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Chat Agent with Vision - UNEG")
        self.geometry("800x700")
        self.resizable(False, False)
        ctk.set_appearance_mode("dark")

        self.current_user = None
        self.chat_history = []
        self.ollama = OllamaClient()

        # Load the vision model
        try:
            self.vision_model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Vision model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading vision model: {e}")
            self.vision_model = None
            messagebox.showerror("Error", "Could not load the vision model. Please check the file 'model_emotions.keras'.")
            self.destroy()
            return
        
        self.load_chat_history()
        self._setup_ui()

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

        self.load_button = ctk.CTkButton(self.top_frame, text="1. Load Image to Identify User", command=self.identify_user_from_image)
        self.load_button.grid(row=0, column=0, pady=10, sticky="ew")

        self.image_label = ctk.CTkLabel(self.top_frame, text="")
        self.image_label.grid(row=1, column=0, pady=5, sticky="ew")
        
        self.info_label = ctk.CTkLabel(self.top_frame, text="Please load an image to start the chat.", font=("Helvetica", 16, "bold"))
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
        
        self.chat_entry = ctk.CTkEntry(self.entry_frame, placeholder_text="Type your message...", height=40)
        self.chat_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.chat_entry.bind("<Return>", self.send_message_event)

        self.send_button = ctk.CTkButton(self.entry_frame, text="Send", command=self.send_message)
        self.send_button.grid(row=0, column=1, sticky="ew")
        
        # Initially, disable chat
        self.disable_chat()

    def identify_user_from_image(self):
        if not self.vision_model:
            messagebox.showerror("Error", "Vision model is not loaded.")
            return

        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        # Display the image
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
            print(f"Warning correcting orientation: {e}")
        display_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(150, 150))
        self.image_label.configure(image=display_image)

        # Process and predict in a separate thread to avoid freezing the GUI
        thread = threading.Thread(target=self.predict, args=(file_path,))
        thread.start()

    def predict(self, file_path):
        try:
            # Preprocess the image
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
                print(f"Warning correcting orientation: {e}")
            # --- Carga compatible con nombres UTF-8 ---
            img_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) == 0:
                self.update_ui_for_user("DESCONOCIDO", "No face detected in the image.")
                return

            x, y, w, h = faces[0]
            face_img = img_array[y:y+h, x:x+w]
            img_resized = cv2.resize(face_img, IMG_SIZE)
            img_normalized = img_resized / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)

            # Make prediction
            predictions = self.vision_model.predict(img_batch)
            emotion_preds, person_preds = predictions[0][0], predictions[1][0]
            
            # Interpret results
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
            self.update_ui_for_user("DESCONOCIDO", f"Error processing image: {e}")
    
    def update_ui_for_user(self, person, emotion):
        self.current_user = person
        prompt_config = PROMPTS.get(person, PROMPTS["DESCONOCIDO"])
        
        info_text = f"User: {person} | Emotion: {emotion}"
        self.info_label.configure(text=info_text)

        # Clear previous chat and set up the new one
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        
        # Initialize chat history with the system message
        self.chat_history = [{"role": "system", "content": prompt_config["system_message"]}]
        
        # Display the welcome message
        self.add_message("Visionary", prompt_config["welcome_message"])
        
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

        self.add_message("You", user_message)
        self.chat_entry.delete(0, "end")

        # Add user message to history for context
        self.chat_history.append({"role": "user", "content": user_message})
        
        # Get AI response in a separate thread
        thread = threading.Thread(target=self.get_ai_response)
        thread.start()

    def get_ai_response(self):
        # Disable input while AI is thinking
        self.chat_entry.configure(state="disabled")
        self.send_button.configure(state="disabled")
        
        response = self.ollama.chat(model='gemma3', messages=self.chat_history)
        
        if response and "message" in response:
            ai_message = response['message']['content']
            
            # Add AI message to history
            self.chat_history.append({"role": "assistant", "content": ai_message})
            self.add_message("Visionary", ai_message)
        else:
            self.add_message("System", "Error: Could not get a response from Ollama. Make sure it is running.")
        
        # Re-enable input
        self.chat_entry.configure(state="normal")
        self.send_button.configure(state="normal")
        self.save_chat_history()

    def add_message(self, sender, message):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"{sender}: {message}\n\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.yview_moveto(1.0) # Auto-scroll

    def enable_chat(self):
        self.chat_entry.configure(state="normal")
        self.send_button.configure(state="normal")

    def disable_chat(self):
        self.chat_entry.configure(state="disabled")
        self.send_button.configure(state="disabled")

    def load_chat_history(self):
        try:
            if os.path.exists(CHAT_DB_FILE):
                with open(CHAT_DB_FILE, 'r') as f:
                    # In a real app, you would load history per user
                    print("Loaded chat history (for reference, not implemented per user).")
        except Exception as e:
            print(f"Could not load chat history: {e}")

    def save_chat_history(self):
        try:
            # Saves the last conversation for the current user
            data_to_save = {
                "user": self.current_user,
                "history": self.chat_history
            }
            with open(CHAT_DB_FILE, 'w') as f:
                json.dump(data_to_save, f, indent=4)
        except Exception as e:
            print(f"Error saving chat history: {e}")


if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()