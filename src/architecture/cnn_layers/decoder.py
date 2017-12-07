from keras.layers.convolutional import Conv2D, UpSampling2D


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
