import app.controller.general_controller as controller
import app.dataset.data as db
import json

def lambda_handler(event):
    try:
        response=controller.general_controller(event)

        return json.dumps(response, ensure_ascii=False, indent=4)
    
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False, indent=4)