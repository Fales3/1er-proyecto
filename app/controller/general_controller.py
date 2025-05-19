import app.dataset.data as data
import app.decorators.exceptions as exceptions

@exceptions.manejador_excepciones
def general_controller(event):
    action=event.pop('action')
    if action=='Análisis total':
        return data.analyze_total()
    if action=='Preprocesamiento':
        df_dict=data.load_original_offendes()
        processed_dict=data.preprocess_data_dict(df_dict, output_dir="data/processed")
        return processed_dict
    if action=='Análisis particular':
        df_dict=data.load_processed_offendes()
        return data.analyze_dict_data(df_dict)
