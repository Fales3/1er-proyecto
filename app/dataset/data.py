import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from typing import Dict, Any

def load_raw_data():
    """Carga los archivos TSV sin modificar."""
    return {
        "train": pd.read_csv("app/OffendES/training_set.tsv", sep="\t"),
        "dev": pd.read_csv("app/OffendES/dev_set.tsv", sep="\t"),
        "test": pd.read_csv("app/OffendES/test_set.tsv", sep="\t")
    }

def load_processed_data():
    """Carga los archivos TSV procesados."""
    return {
        "train": pd.read_csv("data/processed/train_processed.csv"),
        "dev": pd.read_csv("data/processed/dev_processed.csv"),
        "test": pd.read_csv("data/processed/test_processed.csv")
    }

def analyze_general(df_dict):
    """Genera un reporte general de los datos."""
    for subset, df in df_dict.items():
        df["subset"] = subset
    
    analyze_combined_data(df)
    analyze_distribution(df_dict)

    return {
        "message": "Análisis completado",
    }

def analyze_combined_data(df):

    # 2. Distribución de etiquetas
    print("\nDistribución de etiquetas (%):")
    print(df["label"].value_counts(normalize=True).mul(100).round(2))
    
    # 4. Ejemplos aleatorios globales
    print("\nEjemplos aleatorios (global):")
    for label in df["label"].unique():
        sample = df[df["label"] == label].sample(1)
        print(f"\n- Label '{label}': {sample['comment'].values[0]}")

    plot_global_label_distribution(df)

    response ={
        "message": "Análisis combinado completado",
        "stats": {
            "Total_de datos: ": len(df),
            "Distribución de etiquetas: ": df["label"].value_counts(normalize=True).mul(100).round(2).to_dict()
        }
    }
    return response

def plot_global_label_distribution(df):
    """Grafica la distribución global de etiquetas."""
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(data=df, x="label", hue="label", order=df["label"].value_counts().index, palette="viridis", legend=False)
    plt.title("Distribución de etiquetado")
    plt.xlabel("Etiqueta")
    plt.ylabel("Cantidad")
    
    # Añadir porcentajes en las barras
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 50,
                f"{height/total*100:.1f}%", ha="center")
    
    plt.savefig("global_label_distribution.png")
    plt.show()

def analyze_distribution(df_dict):
    """Genera reporte de distribución"""

    report = {
        "total": {},
        "distribution": {},
        "plots": {}
    }

    for name, df in df_dict.items():
        total = len(df)
        report["total"][name] = total
        
        label_dist = df["label"].value_counts(normalize=True).mul(100).round(2)
        report["distribution"][name] = label_dist.to_dict()
    
    plot_path = "label_distribution.png"
    plot_label_distribution(df_dict, save_path=plot_path)
    plot_label_distribution(df_dict)
    report["plots"]["label_distribution"] = plot_path
    
    response ={
        "message": "Análisis de Distribución Completado",
        "reporte": report,
    }
    return response

def plot_label_distribution(df_dict: Dict[str, pd.DataFrame], save_path: str = None):
    """
    Grafica la distribución de etiquetas por subset.
    """
    plt.figure(figsize=(12, 4))
    for i, (name, df) in enumerate(df_dict.items(), 1):
        plt.subplot(1, 3, i)
        df["label"].value_counts().plot(
            kind="bar", 
            color=["skyblue", "orange", "green", "red"]
        )
        plt.title(f"Distribución en {name}")
        plt.xlabel("Etiqueta")
        plt.ylabel("Count")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def preprocess_data_dict(df_dict, output_dir="data/processed", force_reprocess=False):
    "Procesa los datos para clasificación binaria."
    
    offensive_labels={"OFP", "OFG"}
    processed_dict = {}
    os.makedirs(output_dir, exist_ok=True)

    for subset, df in df_dict.items():
        output_path = Path(output_dir) / f"{subset}_processed.csv"
        
        # Cargar datos procesados si existen y no se fuerza reprocesamiento
        if not force_reprocess and os.path.exists(output_path):
            processed_df = pd.read_csv(output_path)
            print(f"✅ Datos {subset} cargados desde caché: {output_path}")
        else:
            # Procesamiento
            processed_df = df.copy()
            processed_df["label"] = processed_df["label"].apply(
                lambda x: 1 if x in offensive_labels else 0
            )
            processed_df = processed_df[["comment", "label"]]
            
            # Guardar
            processed_df.to_csv(output_path, index=False)
            print(f"✨ Datos {subset} procesados y guardados en: {output_path}")
        
        processed_dict[subset] = processed_df
    
    response={
        "message": "Preprocesamiento completado",
        "processed_files": {subset: str(Path(output_dir) / f"{subset}_processed.csv") for subset in df_dict.keys()}
    }

    return response