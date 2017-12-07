import os

import numpy as np
from PIL import Image
from keras import Model
from keras.engine.topology import Input

from decoder import init_decoder
from encoder import init_encoder
from fusion import init_fusion


def train():
    X_train = load_image_data('../dataset/train', normalize=True)


def load_image_data(folder_path, normalize=True):
    array = np.array(  # Array of all images
        [np.array(
            Image.open(
                # Get full path of image file
                os.path.join(folder_path, file_name)))

            # Repeat this for all files in the folder
            for file_name in os.listdir(folder_path)]
    )

    if normalize:
        # Normalize all pixel values from 0-255 to 0-1.0
        return 1.0 / 255 * array

    return array


def init_model():
    input = Input(shape=(1000,))
    encoder = init_encoder()
    fusion = init_fusion(input=input, encoder=encoder)
    decoder = init_decoder(fusion=fusion)

    return Model(inputs=[encoder, input], outputs=decoder)


if __name__ == '__main__':
    train()

    model = init_model()
