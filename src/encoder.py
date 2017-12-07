from keras.engine.topology import Input
from keras.layers.convolutional import Conv2D

from constants import Constants


def init_encoder():
    # TODO maybe should be (H, W, 1,)
    input = Input(shape=(Constants.Global.H, Constants.Global.W,))

    encoder = add_layer_with_strides(input=input, filters=64)
    encoder = add_layer(input=encoder, filters=128)
    encoder = add_layer_with_strides(input=encoder, filters=128)
    encoder = add_layer(input=encoder, filters=256)
    encoder = add_layer_with_strides(input=encoder, filters=256)
    encoder = add_layer(input=encoder, filters=512)
    encoder = add_layer(input=encoder, filters=512)
    encoder = add_layer(input=encoder, filters=256)

    return encoder


def add_layer_with_strides(input, filters):
    return Conv2D(filters=filters,
                  kernel_size=Constants.Encoder.KERNEL_SIZE,
                  activation=Constants.Encoder.ACTIVATION_RELU,
                  padding=Constants.Encoder.PADDING_SAME,
                  strides=Constants.Encoder.STRIDES)(input)


def add_layer(input, filters):
    return Conv2D(filters=filters,
                  kernel_size=Constants.Encoder.KERNEL_SIZE,
                  activation=Constants.Encoder.ACTIVATION_RELU,
                  padding=Constants.Encoder.PADDING_SAME)(input)
