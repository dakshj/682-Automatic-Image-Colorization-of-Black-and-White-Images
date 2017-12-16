import os

import numpy as np
from skimage.io._io import imread, imsave

os.chdir('..')

print(os.getcwd())

test_dir = os.path.join(os.getcwd(), 'dataset', 'test')
test_copy_dir = os.path.join(os.getcwd(), 'dataset', 'test_copy')

if not os.path.exists(test_copy_dir):
    os.makedirs(test_copy_dir)

for file in os.listdir(test_dir):
    image = imread(os.path.join(test_dir, file))
    image = np.stack((image,) * 3, axis=-1)
    imsave(os.path.join(test_copy_dir, file), image)
