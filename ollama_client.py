import requests
import json

class OllamaClient:
    """
    A client to interact with a local Ollama service.
    """
    def __init__(self, host='http://localhost:11434'):
        """
        Initializes the client with the host of the Ollama API.
        """
        self.host = host
        self.api_url = f"{host}/api/chat"
        print("🤖 Ollama client initialized.")

    def chat(self, model, messages, stream=False):
        """
        Sends a request to the Ollama chat API.

        Args:
            model (str): The name of the model to use (e.g., 'llama3').
            messages (list): A list of message dictionaries, following the Ollama format.
            stream (bool): Whether to stream the response or not.

        Returns:
            dict or iterator: The JSON response from the API or an iterator if streaming.
        """

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            if stream:
                return response.iter_lines()
            else:
                return response.json()

        except requests.exceptions.RequestException as e:
            print(f"❌ Error connecting to Ollama at {self.api_url}: {e}")
            print("➡️ Please ensure that Ollama is running and accessible.")
            return None