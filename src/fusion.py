from keras.engine.topology import Input
from keras.layers.convolutional import Conv2D
from keras.layers.core import RepeatVector, Reshape
from keras.layers.merge import concatenate


def init_fusion(encoder):
    embed = Input(shape=(1000,))
    fusion = RepeatVector(n=32 * 32)(embed)
    fusion = Reshape(target_shape=(32, 32, 1000))(fusion)
    fusion = concatenate(inputs=[encoder, fusion], axis=3)
    fusion = Conv2D(filters=256, kernel_size=(1, 1),
                    activation='relu',
                    padding='same')(fusion)

    return fusion, embed
