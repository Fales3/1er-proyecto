import os
import json
import app.decorators.exceptions as exceptions

@exceptions.manejador_excepciones
def set_environment(file=""):
    """
    Configura el entorno de trabajo y carga las variables de entorno desde un archivo JSON.
    
    Args:
        file (str): Ruta al archivo JSON con las variables de entorno. Por defecto es "".
    
    Returns:
        dict: Diccionario con las variables de entorno cargadas.
    """
    with open(file) as f:
        data = json.load(f)
        if type(data) is not dict:
            data= json.loads(data)

    credentials = data["credentials"]
    
    os.environ["OF_ORIGINAL_PATH"] = credentials["OF_ORIGINAL_PATH"]
    os.environ["OF_PROCESSED_PATH"] = credentials["OF_PROCESSED_PATH"]
    os.environ["IMAGES_PATH"] = credentials["IMAGES_PATH"]
    os.environ["BETO_TOKENIZER"] = credentials["BETO_TOKENIZER"]
    os.environ["TOKENIZE_PATH"] = credentials["TOKENIZE_PATH"]
    