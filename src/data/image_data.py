import os
from os.path import join

import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2gray, gray2rgb, rgb2lab
from skimage.color.colorconv import lab2rgb
from skimage.io import imsave

from architecture.cnn_layers.encoder import IMAGE_HEIGHT, IMAGE_WIDTH
from architecture.inception.inception import extract_inception_features


def load_raw_image_data(images_dir, normalize=True, get_file_names=False):
    array = np.array(  # Array of all images
        [np.array(
            load_and_close_image(
                # Get full path of image file
                os.path.join(images_dir, file_name)))

            # Repeat this for all JPEG files in the directory
            for file_name in os.listdir(images_dir) if is_file(file_name)],
        dtype=float
    )

    if normalize:
        # Normalize all pixel values from 0-255 to 0-1.0
        array *= 1.0 / 255

    if get_file_names:
        return array, [file_name for file_name in os.listdir(images_dir) if is_file(file_name)]

    return array


def load_and_close_image(file_path):
    temp = Image.open(file_path)
    image = temp.copy()
    temp.close()
    return image


def is_file(file_name):
    return file_name.endswith('.jpg') or file_name.endswith('.jpeg')


def generate_augmented_image_data(X_train, batch_size=32):
    for images in ImageDataGenerator(
            rotation_range=20,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
    ).flow(x=X_train, batch_size=batch_size):
        yield get_channel_data_from_raw_image_data(images)


def get_channel_data_from_raw_image_data(images, return_a_b_channels_data=True):
    # Make the images gray. Current dimension of each image is H x W
    gray_images = rgb2gray(images)

    # Bring the dimensions of the images back to H x W x 3
    gray_images = gray2rgb(gray_images)

    # Convert to LAB space
    if return_a_b_channels_data:
        lab_images = rgb2lab(images)
    else:
        # Use the grayed images for getting LAB data
        lab_images = rgb2lab(gray_images)

    # Pick the "L" channel from all images.
    # This is going to be the second part of our Training Data.
    l_channel_data = lab_images[:, :, :, 0]  # Dimensions = No. of images, Height, Width, Channel

    # Reshape to add "1" to the dimension of the L channel images
    # Reshapes from (No., H, W) to (No., H, W, 1)
    l_channel_data = l_channel_data.reshape(l_channel_data.shape + (1,))

    # Training data is a combination of:
    # - The L channel
    # - An extraction of high-level feature embeddings from Inception ResNet v2
    gray_data = [l_channel_data, extract_inception_features(gray_images)]

    if return_a_b_channels_data:
        # Pick the "A" and "B" channels from all images. This is gong to be our Ground Truth.
        # Normalize the channels by dividing by 128
        a_b_channels_data = lab_images[:, :, :, 1:] / 128

        # Return the training data and ground truth
        return gray_data, a_b_channels_data

    # Only return training data, as ground truth is not required
    return gray_data


def reconstruct_image_data_from_channels_and_save_images_to_disk(l_channel_data, a_b_channels_data,
                                                                 colorized_dir, test_file_names):
    if l_channel_data.shape[0] != a_b_channels_data.shape[0]:
        raise ValueError('Differing lengths in "l_channel_data"=%s and "a_b_channels_data"=%s'
                         % (l_channel_data.shape[0], a_b_channels_data.shape[0]))

    images = np.zeros((len(l_channel_data), IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    # Set L channel
    images[:, :, :, 0] = l_channel_data[:, :, :, 0]

    # Set A and B channels
    images[:, :, :, 1:] = a_b_channels_data

    if not os.path.exists(colorized_dir):
        os.makedirs(colorized_dir)

    for i, image in enumerate(images):
        # Convert from LAB color space to RGB color space
        image = lab2rgb(lab=image)

        # Save RGB image to disk
        imsave(fname=join(colorized_dir, test_file_names[i]), arr=image)
