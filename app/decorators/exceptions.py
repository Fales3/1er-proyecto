import logging
import traceback

# Configurar el logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

def manejador_excepciones(func):
    """Decorador para manejar excepciones y mostrar información detallada."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_trace = traceback.format_exc()  # Captura la traza del error
            logging.error(f"Error en {func.__name__}: {e}\n{error_trace}")  
            return None
    return wrapper

