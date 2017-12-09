import math
import os
import time
from os.path import join

import numpy as np
import tensorflow as tf
from PIL import Image
from keras import Model
from keras.applications import inception_resnet_v2, InceptionResNetV2
from keras.callbacks import TensorBoard
from keras.engine.topology import Input
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import RepeatVector, Reshape
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2gray, gray2rgb, rgb2lab
from skimage.transform import resize


def load_raw_image_data(images_dir, normalize=True):
    array = np.array(  # Array of all images
        [np.array(
            Image.open(
                # Get full path of image file
                os.path.join(images_dir, file_name)))

            # Repeat this for all JPEG files in the directory
            for file_name in os.listdir(images_dir)
            if file_name.endswith('.jpg') or file_name.endswith('.jpeg')]
    )

    if normalize:
        # Normalize all pixel values from 0-255 to 0-1.0
        return 1.0 / 255 * array

    return array


def generate_image_data_for_inception(X_train, batch_size=32):
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
    lab_images = rgb2lab(images)

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


def reconstruct_image_data_from_channels(l_channel_data, a_b_channels_data):
    images = []

    for i in range(len(l_channel_data)):
        image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))

        # Set L channel
        image[:, :, 0] = l_channel_data[i][:, :, 0]

        # Set A and B channels
        image[:, :, 1:] = a_b_channels_data[i]

        # Save image
        images.append(image)

    return images


def save_image_data_as_images(image_data, colorized_dir):
    for i, image_array in enumerate(image_data):
        Image.fromarray(image_array.astype('uint8'), 'RGB').save(
            os.path.join(colorized_dir, '%s.jpg' % i)
        )


##############
##############
##############

FILE_MODEL_WEIGHTS = 'model_weights.h5'
FILE_MODEL_JSON = 'model.json'


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
    # Save Model Weights
    model.save_weights(join(model_dir, FILE_MODEL_WEIGHTS))

    # Save Model as Serialized JSON
    with open(file=join(model_dir, FILE_MODEL_JSON), mode='w') as json:
        json.write(model.to_json())


##############
##############
##############


def init_tensorboard_for_logging(tensorboard_log_dir):
    return TensorBoard(log_dir=tensorboard_log_dir)


##############
##############
##############


def init_decoder(fusion):
    decoder = add_conv_layer(input=fusion, filters=128)
    decoder = add_upsampling_layer(input=decoder)
    decoder = add_conv_layer(input=decoder, filters=64)
    decoder = add_upsampling_layer(input=decoder)
    decoder = add_conv_layer(input=decoder, filters=32)
    decoder = add_conv_layer(input=decoder, filters=16)
    decoder = add_conv_layer(input=decoder, filters=2, activation='tanh')
    decoder = add_upsampling_layer(input=decoder)

    return decoder


def add_conv_layer(input, filters, kernel_size=(3, 3),
                   activation='relu'):
    return Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  activation=activation,
                  padding='same')(input)


def add_upsampling_layer(input):
    return UpSampling2D(size=(2, 2))(input)


##############
##############
##############

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


def init_encoder():
    input = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1,))

    encoder = add_conv_layer_with_strides(input=input, filters=64)
    encoder = add_conv_layer(input=encoder, filters=128)
    encoder = add_conv_layer_with_strides(input=encoder, filters=128)
    encoder = add_conv_layer(input=encoder, filters=256)
    encoder = add_conv_layer_with_strides(input=encoder, filters=256)
    encoder = add_conv_layer(input=encoder, filters=512)
    encoder = add_conv_layer(input=encoder, filters=512)
    encoder = add_conv_layer(input=encoder, filters=256)

    return input, encoder


def add_conv_layer_with_strides(input, filters):
    return Conv2D(filters=filters,
                  kernel_size=(3, 3),
                  activation='relu',
                  padding='same',
                  strides=2)(input)


def add_conv_layer(input, filters):
    return Conv2D(filters=filters,
                  kernel_size=(3, 3),
                  activation='relu',
                  padding='same')(input)


##############
##############
##############


def init_fusion(input, encoder):
    fusion = RepeatVector(n=32 * 32)(input)
    fusion = Reshape(target_shape=(32, 32, 1000))(fusion)
    fusion = concatenate(inputs=[encoder, fusion], axis=3)
    fusion = Conv2D(filters=256, kernel_size=(1, 1),
                    activation='relu',
                    padding='same')(fusion)

    return fusion


##############
##############
##############


def get_inception_initialized_with_imagenet_weights():
    # Inception ResNet v2 uses weights of the model pre-trained on ImageNet by default
    inception = InceptionResNetV2()

    # Get the default TensorFlow graph being used in the current thread
    inception.graph = tf.get_default_graph()

    return inception


def extract_inception_features(gray_images,
                               inception=get_inception_initialized_with_imagenet_weights()):
    # Preprocess the pixel data to be compatible with Inception ResNet v2
    gray_images_resized = inception_resnet_v2.preprocess_input(
        np.array(
            # Resize each image to 299 x 299 x 3 (as per Inception ResNet v2's specifications)
            # (Need to specify mode because it throws a deprecation warning otherwise)
            [resize(gray_image, (299, 299, 3), mode='constant')
             for gray_image in gray_images]
        )
    )

    with inception.graph.as_default():
        predict = inception.predict(gray_images_resized)

    return predict


##############
##############
##############

def init_model():
    fusion_input = Input(shape=(1000,))
    encoder_input, encoder = init_encoder()
    fusion = init_fusion(input=fusion_input, encoder=encoder)
    decoder = init_decoder(fusion=fusion)

    return Model(inputs=[encoder_input, fusion_input], outputs=decoder)


def train(train_dir, log_dir, epochs, batch_size=32):
    X_train = load_raw_image_data(train_dir)
    model = init_model()
    model.compile(optimizer='adam', loss='mse')

    steps_per_epoch = math.ceil(len(X_train) / batch_size)

    print('X_train length = %s' % len(X_train))
    print('batch size = %s' % batch_size)
    print('steps per epoch = %s' % steps_per_epoch)

    # Fit data using the ImageDataGenerator
    model.fit_generator(
        generator=generate_image_data_for_inception(X_train=X_train, batch_size=batch_size),
        callbacks=[init_tensorboard_for_logging(log_dir)],
        epochs=epochs,
        steps_per_epoch=steps_per_epoch
    )

    return model


def test(model, test_dir, colorized_dir):
    gray_data = load_raw_image_data(test_dir)
    gray_data = get_channel_data_from_raw_image_data(gray_data, return_a_b_channels_data=False)

    # Multiply the predicted values by 128 to convert them to the 0-255 space
    # (i.e., the reverse of what is done when getting the A and B channels from the original images)
    predicted_a_b_channels_data = model.predict(gray_data) * 128

    l_channel_data, _ = gray_data

    final_images = reconstruct_image_data_from_channels(
        l_channel_data=l_channel_data,
        a_b_channels_data=predicted_a_b_channels_data
    )

    save_image_data_as_images(image_data=final_images, colorized_dir=colorized_dir)


if __name__ == '__main__':
    # Go to project root directory
    os.chdir('..')
    os.chdir('..')

    train_dir, log_dir, model_dir, test_dir, colorized_dir = \
        get_project_dirs(project_root_dir=os.getcwd())

    model = train(train_dir=train_dir, log_dir=log_dir, epochs=20, batch_size=32)

    save_model_to_disk(model=model, model_dir=model_dir)

    test(model, test_dir=test_dir, colorized_dir=colorized_dir)
