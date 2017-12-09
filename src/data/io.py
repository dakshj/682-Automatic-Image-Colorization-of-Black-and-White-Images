import os
import time
from os.path import join

from keras.models import load_model

FILE_SAVED_MODEL = 'model.h5'


def get_project_dirs(project_root_dir):
    dataset_dir = join(project_root_dir, 'dataset')

    train_dir = join(dataset_dir, 'train')
    log_dir = join(project_root_dir, 'logs')
    model_dir = join(project_root_dir, 'model')
    test_dir = join(dataset_dir, 'test')
    colorized_dir = join(dataset_dir, 'colorized-' + time.strftime("%Y%m%d-%H%M%S"))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    os.makedirs(colorized_dir)

    return train_dir, log_dir, model_dir, test_dir, colorized_dir


def save_model_to_disk(model, model_dir):
    model.save(filepath=join(model_dir, FILE_SAVED_MODEL))


def load_model_from_disk(model_dir):
    return load_model(join(model_dir, FILE_SAVED_MODEL))
