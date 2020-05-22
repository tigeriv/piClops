import pickle
import pandas as pd
import numpy as np
import glob
import re
import imageio
import cv2
import matplotlib.pyplot as plt
import random
import skimage


cat_to_color = {0: (0, 0, 0), 1: (0, 0, 255)}


def display_image(image):
    plt.imshow(image)
    plt.show()


def cat_to_im(image):
    new_image = np.zeros((len(image), len(image[0]), 3), dtype=np.int32)
    for row_num in range(len(image)):
        for ind_num in range(len(image[row_num])):
            category = image[row_num][ind_num]
            new_image[row_num, ind_num] = cat_to_color[category]
    return new_image


def extract_file_name(file):
    return re.search(r'\d+', file).group(0)


def data_extension(file_name, train=True):
    if train:
        return './TapeImages/image' + str(file_name) + '.jpg'
    else:
        return './MaskImages/' + str(file_name) + '.npy'


def load_file(file):
    file_name = extract_file_name(file)
    img = imageio.imread(data_extension(file_name, train=True))
    mask = np.load(data_extension(file_name, train=False))
    return img, mask


def augment_image(img, mask):
    flip = random.choice([-1, 0, 1])
    augmented_img = cv2.flip(img, flip)
    augmented_mask = cv2.flip(mask, flip)
    kernel_size = random.choice([1, 3, 5, 7])
    augmented_img = cv2.GaussianBlur(augmented_img, (kernel_size, kernel_size), 0)
    return augmented_img, augmented_mask


def get_batch(batch_files):
    batch_img = []
    batch_mask = []
    for file in batch_files:
        img, mask = load_file(file)
        augmented_img, augmented_mask = augment_image(img, mask)
        batch_img.append(augmented_img)
        batch_mask.append(augmented_mask)
    return np.asarray(batch_img), np.asarray(batch_mask)


class TapeRoad:
    def __init__(self, train_percent=0.90):
        self.files = self.get_files()
        self.train_percent = train_percent
        self.train_files, self.val_files = self.split_files()
        self.N = len(self.train_files)
        self.pos = 0
        self.EndOfData = False

    def split_files(self):
        self.shuffle_data()
        split_index = int(self.train_percent * len(self.files))
        train_files = self.files[:split_index]
        val_files = self.files[split_index:]
        return train_files, val_files

    def get_files(self):
        # Obtain train and test data
        file_names = []
        for file in glob.glob('./MaskImages/*'):
            file_names.append(file)
        return file_names

    def shuffle_data(self):
        self.pos = 0
        np.random.shuffle(self.files)
        self.EndOfData = False

    def get_batch(self, batch_size=2):
        batch_files = self.train_files[self.pos: self.pos+batch_size]
        self.pos += batch_size
        if self.pos >= len(self.train_files):
            self.EndOfData = True
        batch_x, batch_y = get_batch(batch_files)
        return batch_x, batch_y

    def get_val_data(self, batch_size=2):
        np.random.shuffle(self.val_files)
        val_batch = self.val_files[:batch_size]
        val_x, val_y = get_batch(val_batch)
        return val_x, val_y

    def reset_pos(self):
        self.pos = 0


# Test the class
if __name__ == "__main__":
    data = TapeRoad()
    batch_x, batch_y = data.get_batch()
    display_image(batch_x[0])
    display_image(cat_to_im(batch_y[0]))