import os

import numpy as np
from keras.preprocessing.image import img_to_array, load_img

from encoder import init_encoder


def train():
    X_train = load_image_data('../dataset/train', normalize=True)


def load_image_data(folder_path, normalize=True):
    array = np.array(  # Array of all images
        [np.array(

            # Use keras to load image as an array
            img_to_array(load_img(

                # Get full path of image file
                os.path.join(folder_path, file_name)
            )))

            # Repeat this for all files in the folder
            for file_name in os.listdir(folder_path)])

    if normalize:
        # Normalize all pixel values from 0-255 to 0-1.0
        return 1.0 / 255 * array

    return array


if __name__ == '__main__':
    train()

    init_encoder()
