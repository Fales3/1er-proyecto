import app.dataset.data as data
import app.decorators.exceptions as exceptions
import app.transformers.tokenizer as tok

@exceptions.manejador_excepciones
def general_controller(event):
    action=event.pop('action')
    if action=='Análisis total':
        return data.analyze_total()
    if action=='Preprocesamiento':
        return data.preprocess_data_dict()
    if action=='Análisis particular':
        df_dict=data.load_processed_offendes()
        return data.analyze_dict_data(df_dict)
    if action=='tokenizar':
        df_dict=data.load_processed_offendes()
        return tok.save_tokenized_data(df_dict, 512)
