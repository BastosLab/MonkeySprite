import os

import cv2 as cv
import numpy as np
from skimage.transform import resize

def generate_simiansprites(patch_size, anti_aliasing=True):
    simian_files = ['banana_09s.jpg', 'cucumber_08s.jpg', 'grape_14s.jpg',
                    'strawberry_04n.jpg']
    sprites = []
    attr = {'labels': []}
    for path in simian_files:
        fruit = path.split('_')[0]
        path = os.path.join('_data', 'simiansprite', path)
        img = cv.imread(path)
        assert img.shape == (280, 280, 3)

        # Normalize
        img = np.float32(img) / 255
        if patch_size != 280:
            img = resize(img, (patch_size, patch_size, 3), order=3,
                         mode='constant', anti_aliasing=anti_aliasing)
        # Back to [0, 255] uint8
        img = (img * 255).astype('uint8')

        sprites.append(img)
        attr['labels'].append(fruit)

    return np.stack(sprites, axis=0), attr
