import app.dataset.data as data
from transformers import AutoTokenizer


def tokenize_function(examples):
    """
    Inicializa el tokenizador y lo aplica a los datos de entrada.
    """
    # Cargar el tokenizador
    tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

    # Tokenizar los datos de entrada
    return tokenizer(
        examples["comment"],
        truncation=True,      # Cortar textos largos
        padding="max_length", # Rellenar con ceros hasta longitud máxima
        max_length=128,       # Máximo de tokens por texto
        return_tensors="np"   # Retornar arrays numpy (para eficiencia)
    )

    return tokenized_data

def tokenize_datasets():
    """
    Tokeniza los conjuntos de datos de entrada.
    """
    # Cargar los conjuntos de datos
    data.load_data()

    # Tokenizar los conjuntos de datos
    tokenized_datasets = {
    subset: dataset.map(tokenize_function, batched=True)
    for subset, dataset in data.processed_dict.items()
    }
    
