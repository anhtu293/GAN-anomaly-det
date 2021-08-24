import tensorflow as tf
import os
import numpy as np
import math
import cv2
import random
from skimage.util import random_noise


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, path, batch_size, img_size, **kwargs):
        super(DataGenerator, self).__init__(**kwargs)
        self.path = path
        self.files = os.listdir(path)
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return math.ceil(len(self.files) / self.batch_size)

    def __load_image(self, file):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        # img = (img - 127.5) / 127.5
        img = img / 255.
        return img.astype(np.float32)

    def __add_noise(self, imgs):
        return random_noise(imgs)

    def shuffle(self):
        random.shuffle(self.files)

    def __getitem__(self, index):
        names = self.files[index * self.batch_size:(index + 1) * self.batch_size]
        org_imgs = [self.__load_image(os.path.join(self.path, name)) 
                    for name in names]
        org_imgs = np.asarray(org_imgs)
        noisy_imgs = self.__add_noise(org_imgs)
        return org_imgs, noisy_imgs
