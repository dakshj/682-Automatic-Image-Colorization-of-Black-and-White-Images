from keras.preprocessing.image import ImageDataGenerator
from skimage.color.colorconv import rgb2gray, gray2rgb, rgb2lab


def generate_training_images_data(X_train):
    for images in ImageDataGenerator(
            rotation_range=20,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
    ).flow(x=X_train):
        # Make the images gray
        gray_images = rgb2gray(images)

        # Bring the dimensions of the images back to H x W x 3
        gray_images = gray2rgb(gray_images)

        # Convert to LAB space
        lab_images = rgb2lab(images)

        # Pick the "L" channel from all images.
        # This is going to be the second part of our Training Data.
        X_train_l = lab_images[:, :, :, 0]  # Dimensions = No. of images, Height, Width, Channel

        # Reshape to add "1" to the dimension of the L channel images
        # Reshapes from (No., H, W) to (No., H, W, 1)
        X_train_l = X_train_l.reshape(X_train_l.shape + (1,))

        # Pick the "A" and "B" channels from all images. This is gong to be our Ground Truth.
        # Normalize the channels by dividing by 128
        Y_a_b = lab_images[:, :, :, 1:] / 128

        # Training data is a combination of the L channel and TODO Inception embedding
        training_data = X_train_l, None  # TODO replace with Inception Embedding

        # Yield the generated images
        yield training_data, Y_a_b
