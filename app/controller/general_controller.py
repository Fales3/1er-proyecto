import app.analyze_dataset.data_functions as data_f
import app.decorators.exceptions as exceptions
import app.model_transformer.tokenizer_functions as tok
import app.model_transformer.design_transformer as design_trans

@exceptions.manejador_excepciones
def general_controller(event):
    action=event.pop('action')
    if action=='Análisis total':
        return data_f.analyze_total()
    if action=='Preprocesamiento':
        return data_f.preprocess_data_dict()
    if action=='Análisis particular':
        df_dict=data_f.load_processed_offendes()
        return data_f.analyze_dict_data(df_dict)
    if action=='tokenizar':
        df_dict=data_f.load_processed_offendes()
        return tok.save_tokenized_data(df_dict, 512)
    if action=='ejemplo tokenizado':
        return tok.show_tokenized_example()
    if action=='entrenar modelo':
        return design_trans.train_model()
