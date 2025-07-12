import requests
import json

class OllamaClient:
    """
    Cliente para interactuar con el servicio local de Ollama.
    """
    def __init__(self, host='http://localhost:11434'):
        """
        Inicializa el cliente con el host de la API de Ollama.
        """
        self.host = host
        self.api_url = f"{host}/api/chat"
        print("🤖 Cliente de Ollama inicializado.")

    def chat(self, model, messages, stream=False):
        """
        Envía una solicitud a la API de chat de Ollama.

        Args:
            model (str): Nombre del modelo a usar (ejemplo: 'llama3').
            messages (list): Lista de diccionarios de mensajes, siguiendo el formato de Ollama.
            stream (bool): Si se debe recibir la respuesta en streaming o no.

        Returns:
            dict o iterator: La respuesta JSON de la API o un iterador si es streaming.
        """

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status() # Lanza una excepción para códigos de estado erróneos (4xx o 5xx)

            if stream:
                return response.iter_lines()
            else:
                return response.json()

        except requests.exceptions.RequestException as e:
            print(f"❌ Error al conectar con Ollama en {self.api_url}: {e}")
            print("➡️ Por favor, asegúrate de que Ollama esté ejecutándose y sea accesible.")
            return None