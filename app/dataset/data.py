import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from typing import Dict, Any

def analyze_total():
    """
    Función para analizar el total de datos.
    """

    # Cargar los datos originales
    df_dict=load_original_offendes()
    
    # Generar un análisis general
    for subset, df in df_dict.items():
        df["subset"] = subset
    
    df_dict_processed=load_processed_offendes()

    stats = {
        "Original": analyze_data(pd.concat(df_dict.values()), "Análisis general.png"),
        "Distribuido": analyze_dict_data(df_dict, "Análisis distribuido.png"),
        "Procesado": analyze_dict_data(df_dict_processed, "Análisis procesado.png"),
    }

    return {"message": "Análisis completado",
            "Datasets": stats}

def load_original_offendes():
    """Carga los archivos TSV."""
    return {
        "train": pd.read_csv(os.environ.get("OF_ORIGINAL_PATH") + "training_set.tsv", sep="\t").copy(),
        "dev": pd.read_csv(os.environ.get("OF_ORIGINAL_PATH") + "dev_set.tsv", sep="\t").copy(),
        "test": pd.read_csv(os.environ.get("OF_ORIGINAL_PATH") + "test_set.tsv", sep="\t").copy()
    }

def analyze_data(df, save_path: str = "img.png"):
    """Genera un análisis combinado de los datos."""
    
    examples={}
    for label in df["label"].unique():
        sample = df[df["label"] == label].sample(1)
        examples[label] = sample['comment'].values[0]

    plot_data_distribution(df, save_path)

    response ={
        "Total de datos: ": len(df),
        "Distribución de etiquetas (%): ": df["label"].value_counts(normalize=True).mul(100).round(2).to_dict(),
        "Ejemplos aleatorios: ": examples,
        "Gráfico guardado en: ": os.path.join(os.environ.get("IMAGES_PATH", ""), save_path)
    }
    return response

def plot_data_distribution(df, save_path: str = "img.png"):
    """Grafica la distribución global de etiquetas."""
    plt.figure(figsize=(10, 5))
    ax = df["label"].value_counts().sort_values(ascending=False).plot(
        kind="bar",
        color=["skyblue", "orange", "green", "red"],
        legend=False
    )
    plt.title("Distribución de etiquetado")
    plt.xlabel("Etiqueta")
    plt.ylabel("Cantidad")
    
    # Añadir porcentajes en las barras
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 50,
                f"{height/total*100:.1f}%", ha="center")
    
    if save_path:
        Path(os.environ.get("IMAGES_PATH")).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.environ.get("IMAGES_PATH") + save_path)
    plt.show()

def analyze_dict_data(df_dict, save_path: str = "img.png"):
    """Genera reporte de distribución"""

    response = {
        "Cantidad por subset": {},
        "Distribución (%)": {},
        "Gráfico guardado en": {}
    }

    for name, df in df_dict.items():
        total = len(df)
        response["Cantidad por subset"][name] = total
        
        label_dist = df["label"].value_counts(normalize=True).mul(100).round(2)
        response["Distribución (%)"][name] = label_dist.to_dict()
    
    plot_datadict_distribution(df_dict, save_path)
    response["Gráfico guardado en"]["Análisis Distribuido"] = os.path.join(os.environ.get("IMAGES_PATH", ""), save_path)

    return response

def plot_datadict_distribution(df_dict, save_path: str = "img.png"):
    """
    Grafica la distribución de etiquetas por subset.
    """
    plt.figure(figsize=(12, 4))
    for i, (name, df) in enumerate(df_dict.items(), 1):
        plt.subplot(1, 3, i)
        
        ax = df["label"].value_counts().plot(
            kind="bar", 
            color=["skyblue", "orange", "green", "red"]
        )
        plt.title(f"Distribución en {name}")
        plt.xlabel("Etiqueta")
        plt.ylabel("Cantidad")

        total = len(df)
        for p in ax.patches:
            height = p.get_height()
            ax.text(
                p.get_x() + p.get_width() / 2.,
                height + 5,  # Ajusta este valor para separar el texto de la barra
                f"{height/total*100:.1f}%",
                ha="center",
                fontsize=10
            )
    
    plt.tight_layout()
    if save_path:
        Path(os.environ.get("IMAGES_PATH")).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.environ.get("IMAGES_PATH") + save_path)
    plt.show()

def load_processed_offendes():
    """Carga los archivos TSV procesados."""
    return {
        "train": pd.read_csv(os.environ.get("OF_PROCESSED_PATH")+"train_processed.csv").copy(),
        "dev": pd.read_csv(os.environ.get("OF_PROCESSED_PATH")+"dev_processed.csv").copy(),
        "test": pd.read_csv(os.environ.get("OF_PROCESSED_PATH")+"test_processed.csv").copy()
    }


def preprocess_data_dict():
    "Procesa los datos para clasificación binaria."
    
    df_dict=load_original_offendes()

    offensive_labels={"OFP", "OFG"}
    processed_dict = {}
    Path(os.environ.get("OF_PROCESSED_PATH")).mkdir(parents=True, exist_ok=True)

    for subset, df in df_dict.items():
        output_path = Path(os.environ.get("OF_PROCESSED_PATH")) / f"{subset}_processed.csv"
        
        # Cargar datos procesados si existen
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
        else:
            # Procesamiento
            df["label"] = df["label"].apply(
                lambda x: 1 if x in offensive_labels else 0
            )
            df = df[["comment", "label"]]
            
            # Guardar
            df.to_csv(output_path, index=False)
        
        processed_dict[subset] = df
    
    response={
        "Archivos Procesados": {subset: os.path.join(os.environ.get("OF_PROCESSED_PATH") / f"{subset}_processed.csv") for subset in df_dict.keys()}
    }

    return response