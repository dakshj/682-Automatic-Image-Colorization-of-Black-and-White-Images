from os.path import join

FILE_MODEL_WEIGHTS = 'model_weights.h5'
FILE_MODEL_JSON = 'model.json'


def save_model_to_disk(model, folder_path):
    # Save Model Weights
    model.save_weights(join(folder_path, FILE_MODEL_WEIGHTS))

    # Save Model as Serialized JSON
    with open(file=join(folder_path, FILE_MODEL_JSON), mode='w') as json:
        json.write(model.to_json())
