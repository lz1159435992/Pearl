import json


def model_to_dict(model):
    result = {}
    for var in model:
        result[str(var)] = str(model[var])
    return result
def load_dictionary(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)