import numpy as np
import tensorflow as tf
from keras.applications import inception_resnet_v2, InceptionResNetV2
from skimage.transform import resize


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
