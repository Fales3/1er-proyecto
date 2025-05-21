import app.dataset.data as data
from transformers import AutoTokenizer

import os
from pathlib import Path
from datasets import Dataset, DatasetDict
import numpy as np

def load_tokenizer():
    """
    Carga el tokenizador desde el modelo preentrenado.
    """
    # Cargar el tokenizador
    tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

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
    example_data = None

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

        if subset == "train":
            example_idx = 0
            example_data = {
                "texto_original": df["comment"].iloc[example_idx],
                "texto_tokenizado": tokenizer.convert_ids_to_tokens(tokenized["input_ids"][example_idx]),
                "input_ids": tokenized["input_ids"][example_idx].tolist(),
                "attention_mask": tokenized["attention_mask"][example_idx].tolist(),
                "etiqueta": int(df["label"].iloc[example_idx])
            }
    
    tokenize_path = Path(os.environ.get("TOKENIZE_PATH"))
    tokenize_path.mkdir(parents=True, exist_ok=True)
    hf_datasets.save_to_disk(tokenize_path)

    return {
        "message": "Tokenización de la data exitosa",
        "ruta de guardado": str(tokenize_path),
        "ejemplo": {
            "texto_original": example_data["texto_original"],
            "texto_tokenizado": example_data["texto_tokenizado"][:5],
            "input_ids": example_data["input_ids"][:5],
            "attention_mask": example_data["attention_mask"][:5],
            "etiqueta": example_data["etiqueta"]
        },
    }