# train_model.py

import os
import json
import cv2  # Importamos OpenCV
import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split # Para dividir los datos

# --- Configuración del Modelo y Datos ---
# Dimensiones a las que se redimensionarán las imágenes
IMG_HEIGHT = 160
IMG_WIDTH = 160
# Tamaño del lote para el entrenamiento
BATCH_SIZE = 32
# Número de épocas para entrenar el modelo
EPOCHS = 15
# Directorio principal que contiene las carpetas de emociones
DATA_DIR = 'images/'
# Nombre del archivo para guardar el modelo entrenado
MODEL_FILENAME = 'emotion_model.keras'
# Nombre del archivo para guardar las etiquetas de las clases
# CLASSES_FILENAME = 'class_names.json'
# Nombre del archivo para guardar los nombres de las personas
# PERSONS_FILENAME = 'person_names.json'

# Variables globales para las clases y nombres de personas
CLASS_NAMES = []
PERSON_NAMES = []

def setup_hardware():
    """Detecta y configura el hardware (GPU o CPU) para el entrenamiento."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            selected_gpu = gpus[0]
            print(f"✅ Hardware detectado: GPU - {selected_gpu.name}")
            return tf.test.gpu_device_name()
        except RuntimeError as e:
            print(f"⚠️ Error al configurar la GPU: {e}\n▶️ Usando CPU como alternativa.")
            return "/CPU:0"
    else:
        print("▶️ No se detectó GPU. Usando CPU para el entrenamiento.")
        return "/CPU:0"

def load_data_with_opencv():
    """
    Carga, redimensiona y preprocesa imágenes usando OpenCV.
    Detecta y recorta el rostro antes de alimentar al modelo.
    """
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: El directorio de datos '{DATA_DIR}' no existe.")
        return None, None, None

    print("🔄 Cargando y preprocesando imágenes con OpenCV...")

    # Cargar el clasificador Haar Cascade para detección de rostros
    haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    if face_cascade.empty():
        print("❌ Error: No se pudo cargar el clasificador Haar Cascade para rostros.")
        return None, None, None

    images = []
    labels = []
    person_names = []
    
    # Obtener clases y omitir carpetas vacías
    class_names = [d for d in sorted(os.listdir(DATA_DIR)) if os.path.isdir(os.path.join(DATA_DIR, d))]
    valid_class_names = []
    for name in class_names:
        if not os.listdir(os.path.join(DATA_DIR, name)):
            print(f"⚠️ Advertencia: No se encontraron imágenes en la carpeta '{name}'. Se omitirá.")
        else:
            valid_class_names.append(name)
            
    if not valid_class_names:
        print("❌ Error: No se encontraron carpetas con imágenes para entrenar.")
        return None, None, None
        
    class_map = {name: i for i, name in enumerate(valid_class_names)}
    print(f"\nClases detectadas para el entrenamiento: {valid_class_names}")
    # Guardar en variable global
    global CLASS_NAMES
    CLASS_NAMES = valid_class_names

    # Iterar sobre cada carpeta de emoción
    for class_name, label_idx in class_map.items():
        class_path = os.path.join(DATA_DIR, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                # Convertir a escala de grises para la detección de rostros
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Parámetros ajustados para ampliar el rango de detección
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,  # Más sensible (antes 1.1)
                    minNeighbors=2     # Menos estricto (antes 4)
                )
                if len(faces) == 0:
                    print(f"⚠️ Advertencia: No se detectó rostro en '{img_path}'. Se omitirá.")
                    continue
                # Tomar el primer rostro detectado
                (x, y, w, h) = faces[0]
                face_img = img[y:y+h, x:x+w]
                # Redimensionar el rostro detectado
                face_img = cv2.resize(face_img, (IMG_WIDTH, IMG_HEIGHT))
                # Convertir de BGR a RGB
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                images.append(face_img)
                labels.append(label_idx)
                # --- Nuevo: extraer nombre de persona ---
                if img_name.startswith("Magleo_"):
                    person_names.append("Magleo")
                elif img_name.startswith("Hector_"):
                    person_names.append("Hector")
                else:
                    person_names.append("desconocido")

    # Convertir listas a arrays de NumPy
    X = np.array(images)
    y = np.array(labels)
    person_names = np.array(person_names)
    
    # Dividir los datos en conjuntos de entrenamiento y validación
    X_train, X_val, y_train, y_val, person_train, person_val = train_test_split(
        X, y, person_names, test_size=0.3, random_state=42, stratify=y
    )
    # Guardar nombres de personas de validación en variable global
    global PERSON_NAMES
    PERSON_NAMES = list(person_val)

    print(f"📊 Total de imágenes cargadas: {len(X)}")
    print(f"   - Entrenamiento: {len(X_train)} imágenes")
    print(f"   - Validación: {len(X_val)} imágenes")

    # Crear datasets de TensorFlow
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    
    # Optimizar el rendimiento
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(buffer_size=len(X_train)).batch(BATCH_SIZE).cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).cache().prefetch(buffer_size=AUTOTUNE)
    
    # Eliminar guardado de archivos JSON
    # return train_ds, val_ds, valid_class_names
    return train_ds, val_ds

def build_model(num_classes):
    """Construye y compila el modelo CNN."""
    model = models.Sequential([
        # La normalización se hace aquí. Las imágenes ya están redimensionadas.
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    """Función principal para ejecutar el proceso de entrenamiento."""
    device = setup_hardware()
    
    with tf.device(device):
        train_ds, val_ds = load_data_with_opencv()
        
        if train_ds is None:
            return

        model = build_model(num_classes=len(CLASS_NAMES))
        
        print("\n📄 Resumen del Modelo:")
        model.summary()
        
        print(f"\n🚀 Iniciando entrenamiento en {device} por {EPOCHS} épocas...")
        model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
        
        print("\n✅ Entrenamiento completado.")
        model.save(MODEL_FILENAME)
        print(f"💾 Modelo guardado exitosamente como '{MODEL_FILENAME}'")

if __name__ == '__main__':
    main()