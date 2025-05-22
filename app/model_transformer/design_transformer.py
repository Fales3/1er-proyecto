import app.model_transformer.tokenizer_functions as tok

import transformers
import os
from pathlib import Path
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def load_model():
    """
    Carga el modelo desde el modelo preentrenado.
    """
    # Cargar el modelo
    model = AutoModelForSequenceClassification.from_pretrained(os.environ.get("TRANSFORMER_MODEL"),
    num_labels=2  # 0: No ofensivo, 1: Ofensivo
)

    return model

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"precision": precision, "recall": recall, "f1": f1}

def config_trainer():
    """
    Configura el entrenador para el modelo.
    """

    model=load_model()
    hf_datasets = tok.load_tokenized_data()

    base_dir = Path(os.environ.get("TRANSFORMER_PATH"))
    base_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(base_dir / "results"),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(base_dir / "logs")
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_datasets["train"],
        eval_dataset=hf_datasets["dev"],
        compute_metrics=compute_metrics,
    )

    return trainer

def train_model():
    """
    Ejecuta el entrenamiento y guarda el modelo con mejor F1-score.
    """

    trainer=config_trainer()

    trainer.train()
    
    best_model_path = Path(os.environ.get("TRANSFORMER_PATH")) / "best_model"
    trainer.save_model(best_model_path)
    
    return {
        "message": "Modelo entrenado y guardado con éxito.",
        "ruta del modelo": str(best_model_path)
    }