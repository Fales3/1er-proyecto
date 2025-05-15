import lambda_function
from rich import print

event_general={
    "action":"Análisis total",
}
event_preprocess={
    "action":"Preprocesamiento",
}
event_particular={
    "action":"Análisis particular",
}

response=lambda_function.lambda_handler(event_particular)
print("Response JSON: ", response)