from transformers import AutoTokenizer

import os
from pathlib import Path
from datasets import Dataset, DatasetDict
import numpy as np
import random

def load_tokenizer():
    """
    Carga el tokenizador desde el modelo preentrenado.
    """
    # Cargar el tokenizador
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get("BETO_TOKENIZER"))

    # Verificar si el tokenizador se ha cargado correctamente
    if tokenizer is None:
        raise ValueError("No se pudo cargar el tokenizador.")

    return tokenizer

def save_tokenized_data(df_dict, max_length=None):
    """
    Convierte los datos tokenizados (diccionario de arrays) a DatasetDict de Hugging Face
    y los guarda en disco.
    """

    tokenizer=load_tokenizer()

    hf_datasets = DatasetDict()

    for subset, df in df_dict.items():
        # Tokenizar los comentarios
        tokenized = tokenizer(
            df["comment"].tolist(),
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            return_tensors="np"
        )
        
        # Crear Dataset
        hf_dataset = Dataset.from_dict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": np.array(df["label"])
        })

        hf_datasets[subset] = hf_dataset
    
    tokenize_path = Path(os.environ.get("TOKENIZE_PATH"))
    tokenize_path.mkdir(parents=True, exist_ok=True)
    hf_datasets.save_to_disk(tokenize_path)

    return {
        "message": "Tokenización de la data exitosa",
        "ruta de guardado": str(tokenize_path)
    }

def show_tokenized_example(subset="train", idx=None):
    """
    Muestra los datos tokenizados.
    """
    # Cargar el DatasetDict desde disco
    tokenize_path = Path(os.environ.get("TOKENIZE_PATH"))
    hf_datasets = DatasetDict.load_from_disk(tokenize_path)

    dataset = hf_datasets[subset]
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)

    example = dataset[idx]
    tokenizer = load_tokenizer()
    text = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(example['input_ids'])
    
    return {
        "message": "Ejemplo de dato tokenizado",
        "índice": idx,
        "texto decodificado": text,
        "tokens": tokens[:10],
        "input ids": example['input_ids'][:5],
        "attention mask": example['attention_mask'][:5],
        "etiqueta": example['labels']
    }

def load_tokenized_data():
    """
    Carga el DatasetDict desde disco.
    """
    tokenize_path = Path(os.environ.get("TOKENIZE_PATH"))
    hf_datasets = DatasetDict.load_from_disk(tokenize_path)
    
    return hf_datasets