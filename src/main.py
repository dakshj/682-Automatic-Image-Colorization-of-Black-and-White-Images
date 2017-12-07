import os

import numpy as np
from keras.preprocessing.image import img_to_array, load_img


def get_normalized_training_data(folder_path):
    return np.array([np.array(img_to_array(load_img(os.path.join(folder_path, file_name))))
                     for file_name in os.listdir(folder_path)])


def train():
    X_train = get_normalized_training_data('../dataset/train')
    print(X_train)


if __name__ == '__main__':
    train()
