import matplotlib.pyplot as plt
import glob
import numpy as np
import imageio
import re

cat_to_color = {0: (0, 0, 0), 1: (0, 0, 255)}


def display_image(image):
    plt.imshow(image)
    plt.show()


def tape_mask(image):
    image_mask = np.zeros(image.shape[:2])
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if R[row][col] < 2 * G[row][col] and G[row][col] < 0.8 * B[row][col] and B[row][col] > 80 and B[row][col] < 255:
                image_mask[row][col] = 1
    return image_mask


def cat_to_im(image):
    new_image = np.zeros((len(image), len(image[0]), 3), dtype=np.int32)
    for row_num in range(len(image)):
        for ind_num in range(len(image[row_num])):
            category = image[row_num][ind_num]
            new_image[row_num, ind_num] = cat_to_color[category]
    return new_image


# Folder doesn't include the / for directory
def create_masks(folder):
    done_indices = [int(re.search(r'\d+', file).group()) for file in glob.glob("MaskImages/*")]
    for file in glob.glob(folder + "/*"):
        index = int(re.search(r'\d+', file).group())

        if index in done_indices:
            continue

        print(len(done_indices))
        image = imageio.imread(file)
        display_image(image)
        mask = tape_mask(image)
        display_image(cat_to_im(mask))

        rows_to_kill = int(input("Rows to set as background (list max row):"))
        new_mask = np.zeros(mask.shape)
        new_mask[rows_to_kill:, :] += mask[rows_to_kill:, :]
        display_image(cat_to_im(new_mask))

        np.save("MaskImages/" + str(index), new_mask)
    # [<55% G, <75% B, 100+]


if __name__ == "__main__":
    create_masks("TapeImages")

    # for file in glob.glob("TapeImages/image*"):
    #     index = int(re.search(r'\d+', file).group())
    #     image = imageio.imread(file)
    #     imageio.imwrite("TapeImages/" + str(index) + ".jpg", image)