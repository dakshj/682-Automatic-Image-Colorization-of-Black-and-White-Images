import math
import os

from keras import Model
from keras.engine.topology import Input

from architecture.cnn_layers.decoder import init_decoder
from architecture.cnn_layers.encoder import init_encoder
from architecture.cnn_layers.fusion import init_fusion
from data.image_data import generate_image_data_for_inception, load_raw_image_data, \
    get_channel_data_from_raw_image_data, \
    reconstruct_image_data_from_channels_and_save_images_to_disk
from data.io import get_project_dirs
from data.logging import init_tensorboard_for_logging


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

    reconstruct_image_data_from_channels_and_save_images_to_disk(
        l_channel_data=l_channel_data,
        a_b_channels_data=predicted_a_b_channels_data,
        colorized_dir=colorized_dir
    )


if __name__ == '__main__':
    # Go to project root directory
    os.chdir('..')
    os.chdir('..')

    train_dir, log_dir, model_dir, test_dir, colorized_dir = \
        get_project_dirs(project_root_dir=os.getcwd())

    model = train(train_dir=train_dir, log_dir=log_dir, epochs=20, batch_size=32)

    save_model_to_disk(model=model, model_dir=model_dir)

    test(model, test_dir=test_dir, colorized_dir=colorized_dir)
