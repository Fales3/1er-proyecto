import app.dataset.data as data
import app.decorators.exceptions as exceptions

@exceptions.manejador_excepciones
def general_controller(event):
    action=event.pop('action')
    if action=='Análisis total':
        df_dict=data.load_raw_data()
        return data.analyze_general(df_dict)
    if action=='Preprocesamiento':
        df_dict=data.load_raw_data()
        processed_dict=data.preprocess_data_dict(df_dict, output_dir="data/processed")
        return processed_dict
    if action=='Análisis particular':
        df_dict=data.load_processed_data()
        return data.analyze_distribution(df_dict)
