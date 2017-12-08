import os
from os.path import join

FILE_MODEL_WEIGHTS = 'model_weights.h5'
FILE_MODEL_JSON = 'model.json'


def get_project_dirs(project_root_dir):
    dataset_dir = join(project_root_dir, 'dataset')

    train_dir = join(dataset_dir, 'train')
    log_dir = join(project_root_dir, 'logs')
    model_dir = join(project_root_dir, 'model')
    test_dir = join(dataset_dir, 'test')
    colorized_dir = join(test_dir, 'colorized')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(colorized_dir):
        os.makedirs(colorized_dir)

    return train_dir, log_dir, model_dir, test_dir, colorized_dir


def save_model_to_disk(model, model_dir):
    # Save Model Weights
    model.save_weights(join(model_dir, FILE_MODEL_WEIGHTS))

    # Save Model as Serialized JSON
    with open(file=join(model_dir, FILE_MODEL_JSON), mode='w') as json:
        json.write(model.to_json())
