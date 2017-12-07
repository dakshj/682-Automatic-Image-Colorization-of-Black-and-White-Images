from keras.engine.topology import Input
from keras.layers.convolutional import Conv2D
from keras.layers.core import RepeatVector, Reshape
from keras.layers.merge import concatenate

from constants import Constants


def init_fusion(encoder):
    input = Input(shape=(1000,))
    fusion = RepeatVector(n=32 * 32)(input)
    fusion = Reshape(target_shape=(32, 32, 1000))(fusion)
    fusion = concatenate(inputs=[encoder, fusion], axis=3)
    fusion = Conv2D(filters=256, kernel_size=(1, 1),
                    activation=Constants.Fusion.ACTIVATION_RELU,
                    padding=Constants.Fusion.PADDING_SAME)(fusion)

    return fusion
