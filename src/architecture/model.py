from keras import Model
from keras.engine.topology import Input

from architecture.cnn_layers.decoder import init_decoder
from architecture.cnn_layers.encoder import init_encoder
from architecture.cnn_layers.fusion import init_fusion
from data.image_data import generate_image_data_for_inception, load_raw_image_data, \
    get_channel_data_from_raw_image_data, reconstruct_image_data_from_channels, \
    save_image_data_as_images
from data.logging import init_tensorboard_for_logging
from data.model_io import save_model_to_disk


def init_model():
    input = Input(shape=(1000,))
    encoder = init_encoder()
    fusion = init_fusion(input=input, encoder=encoder)
    decoder = init_decoder(fusion=fusion)

    return Model(inputs=[encoder, input], outputs=decoder)


def train(training_data_dir, log_dir, batch_size=32):
    X_train = load_raw_image_data(training_data_dir)
    model = init_model()
    model.compile(optimizer='adam', loss='mse')

    # Fit data using the ImageDataGenerator
    model.fit_generator(
        generator=generate_image_data_for_inception(X_train=X_train, batch_size=batch_size),
        callbacks=[init_tensorboard_for_logging(log_dir)],
        epochs=1000
    )

    return model


def test(model, test_data_dir):
    gray_data = load_raw_image_data(test_data_dir)
    gray_data = get_channel_data_from_raw_image_data(gray_data, return_a_b_channels_data=False)

    # Multiply the predicted values by 128 to convert them to the 0-255 space
    # (i.e., the reverse of what is done when getting the A and B channels from the original images)
    predicted_a_b_channels_data = model.predict(gray_data) * 128

    l_channel_data, _ = gray_data

    final_images = reconstruct_image_data_from_channels(
        l_channel_data=l_channel_data,
        a_b_channels_data=predicted_a_b_channels_data
    )

    save_image_data_as_images(image_data=final_images,
                              colorized_folder_path='../../../dataset/test/colorized')


if __name__ == '__main__':
    model = train(training_data_dir='../../../dataset/train', log_dir='../../../logs')

    save_model_to_disk(model=model, folder_path='../../../model')

    test(model, test_data_dir='../../../dataset/test')
