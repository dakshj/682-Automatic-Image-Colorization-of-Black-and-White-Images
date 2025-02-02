from keras.engine.topology import Input
from keras.layers.convolutional import Conv2D

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
