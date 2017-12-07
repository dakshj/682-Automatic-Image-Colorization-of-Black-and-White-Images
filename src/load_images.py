import os

import numpy as np
from PIL import Image


def load_images_as_numpy(folder_path):
    return np.array([np.array(Image.open(os.path.join(folder_path, file_name)))
                     for file_name in os.listdir(folder_path)])
