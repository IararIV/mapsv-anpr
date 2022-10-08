from numbers import Number
from pathlib import Path
from typing import List, Optional, Tuple, Text

import cv2
import imutils
import numpy as np

import matplotlib.pyplot as plt


def load_image(path: Path) -> np.array:
    image = cv2.imread(str(path))
    return image


def write_image(path: Path, image: np.array) -> None:
    cv2.imwrite(str(path), image)


def resize_image(image: np.array, width: Number = 1100) -> np.array:
    resized_img = imutils.resize(image, width=width)
    return resized_img


def grayscale_image(image: np.array) -> np.array:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def show_image(image: np.array, image_name: str = "image") -> None:
    plt.figure(figsize=(10, 12))
    plt.title(image_name)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def show_contour_detection(image: np.array, edged_image: np.array, contours_list: List) -> None:
    img_contours = image.copy()
    cv2.drawContours(img_contours, contours_list[0], -1, (0, 255, 0), 3)
    img_contours_30 = image.copy()
    contours = sorted(contours_list[1], key=cv2.contourArea, reverse=True)[:30]
    cv2.drawContours(img_contours_30, contours, -1, (0, 255, 0), 3)
    img_plate = image.copy()
    contours = contours_list[2]
    cv2.drawContours(img_plate, contours, -1, (0, 255, 0), 3)

    fig, ax = plt.subplots(1, 5)
    fig.set_size_inches(20, 10)
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original image")
    ax[1].imshow(edged_image, cmap='gray')
    ax[1].set_title("Canny filter")
    ax[2].imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
    ax[2].set_title("Contours from Canny filter")
    ax[3].imshow(cv2.cvtColor(img_contours_30, cv2.COLOR_BGR2RGB))
    ax[3].set_title("Contours from Canny filter sorted by area (30)")
    ax[4].imshow(cv2.cvtColor(img_plate, cv2.COLOR_BGR2RGB))
    ax[4].set_title("Plate detected")
    plt.show()


def show_keypoints(image: np.array, keypoints: List[List[Number]] = None, title: Optional[Text] = "", figsize: Optional[Tuple[Number]] = (12,4)):
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    ax.imshow(image, cmap="gray")
    left_kp = [kp[0] for kp in keypoints]
    right_kp = [kp[1] for kp in keypoints]
    ax.vlines(x = left_kp, ymin=0, ymax=image.shape[0] - 1, color = 'r')
    ax.vlines(x = right_kp, ymin=0, ymax=image.shape[0] - 1, color = 'g')
    plt.show()


def show_split_chars(chars_list: List[np.array], image_name: str = "image"):
    fig, axs = plt.subplots(2, 4)
    fig.suptitle(image_name)
    for i in range(len(chars_list)):
        axs[i//4][i%4].imshow(chars_list[i], cmap="gray")
    fig.delaxes(axs[1][3])
    plt.show()
