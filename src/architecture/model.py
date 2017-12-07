from keras import Model
from keras.engine.topology import Input

from architecture.cnn_layers.decoder import init_decoder
from architecture.cnn_layers.encoder import init_encoder
from architecture.cnn_layers.fusion import init_fusion


def init_model():
    input = Input(shape=(1000,))
    encoder = init_encoder()
    fusion = init_fusion(input=input, encoder=encoder)
    decoder = init_decoder(fusion=fusion)

    return Model(inputs=[encoder, input], outputs=decoder)
