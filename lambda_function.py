import app.controller.general_controller as controller
import app.dataset.data as db
import json
import app.decorators.exceptions as exceptions
import app.config.enviroment_set as set_env

@exceptions.manejador_excepciones
def lambda_handler(event):
    set_env.set_environment("./envMDW.json")
    
    response=controller.general_controller(event)

    return json.dumps(response, ensure_ascii=False, indent=4)
