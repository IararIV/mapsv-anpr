import cv2
import numpy as np

from utils import grayscale_image


class ImageSegmentation:

    def __init__(self):
        # Remove blue box
        self.hsv_blue_start = np.array([100, 100, 100], np.uint8)
        self.hsv_blue_end = np.array([115, 255, 255], np.uint8)
        self.crop_range = (10, 5)

    def segment(self, image: np.array) -> np.array:
        image_without_blue = self.remove_left_box(image)
        image_crop = self.crop(image_without_blue)
        denoised_image = self.denoise(image_crop)
        image_gray = grayscale_image(denoised_image)
        bin_image = self.otsu_bin(image_gray)
        return bin_image

    def remove_left_box(self, image: np.array) -> np.array:
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(img_hsv, self.hsv_blue_start, self.hsv_blue_end)
        idx = np.argwhere(mask_hsv[:, :50] == 255)[:, 1].max()
        return image[:, idx:]

    def crop(self, image: np.array) -> np.array:
        h, w = self.crop_range
        return image[h:-h, w:-w]

    @staticmethod
    def denoise(image: np.array) -> np.array:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    @staticmethod
    def otsu_bin(image: np.array) -> np.array:
        _, bin_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bin_image
