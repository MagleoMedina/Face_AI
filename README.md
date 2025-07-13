# Visionary_AI

**Visionary_AI** es una aplicación de escritorio que combina visión por computadora y chat inteligente para identificar personas y emociones a partir de imágenes, permitiendo conversaciones personalizadas según el usuario y su estado emocional.

## Características

- Identificación de personas y emociones usando modelos de TensorFlow/Keras.
- Interfaz gráfica moderna con CustomTkinter.
- Chat inteligente impulsado por Ollama (modelo local de lenguaje).
- Historial de chats persistente y gestión de múltiples conversaciones.
- Adaptación dinámica de respuestas según usuario y emoción detectada.

## Requisitos

- Python 3.8+
- Ollama instalado y ejecutándose localmente ([https://ollama.com/](https://ollama.com/))
- Dependencias Python:
  - customtkinter
  - pillow
  - opencv-python
  - numpy
  - tensorflow
  - requests

## Instalación

```bash
pip install customtkinter pillow opencv-python numpy tensorflow requests
```

Asegúrate de tener el archivo del modelo `model_emotions.keras` en la raíz del proyecto.

### Instalar el modelo Gemma3 en Ollama

Este proyecto utiliza el modelo **Gemma3** para el chat inteligente. Para instalarlo en Ollama, ejecuta:

```bash
ollama pull gemma:3b
```

Asegúrate de que Ollama esté corriendo y que el modelo esté disponible antes de iniciar la aplicación.

## Uso

1. Ejecuta Ollama localmente.
```bash
ollama serve
```
2. Inicia la aplicación:

```bash
python main_app.py
```

3. Carga una imagen para identificar al usuario y su emoción.
4. Comienza a chatear con el asistente "Visionary".

## Estructura del Proyecto

- `main_app.py`: Interfaz gráfica y lógica principal.
- `ollama_client.py`: Cliente para interactuar con Ollama.
- `chat_history.json`: Historial persistente de chats.
- `model_emotions.keras`: Modelo de Keras para reconocimiento de emociones y personas. Para obternerlo debe compilar la rama Face_AI

## Créditos

Desarrollado por Magleo y Hector.

---


