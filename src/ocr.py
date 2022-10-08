import pickle
from numbers import Number
from typing import List, Tuple, Any, Text

import annoy as annoy
import cv2
import numpy as np
from PIL import Image


class OCR:

    def __init__(self) -> None:
        # Split characters
        self.kp = 8
        self.thr = 0.95
        self.padding = 0
        self.crop_kp = None
        # Predict character
        self.f = 784

        digit_model = annoy.AnnoyIndex(self.f, 'angular')
        digit_model.load("../models/digits/digit_model.ann")
        with open("../models/digits/digit_model_labels.pickle", 'rb') as handle:
            digit_latent_space = pickle.load(handle)
        self.digit_model = {"model": digit_model,
                            "latent_space": digit_latent_space}

        chars_model = annoy.AnnoyIndex(self.f, 'angular')
        chars_model.load("../models/chars/chars_model.ann")
        with open("../models/chars/chars_model_labels.pickle", 'rb') as handle:
            chars_latent_space = pickle.load(handle)
        self.chars_model = {"model": chars_model,
                            "latent_space": chars_latent_space}

    def predict_char(self, idx: Number, image: np.array) -> Text:
        descriptors = self.get_descriptors(image)
        if idx < 4:
            model = self.digit_model["model"]
            latent_space = self.digit_model["latent_space"]
        else:
            model = self.chars_model["model"]
            latent_space = self.chars_model["latent_space"]
        nearest_idx = model.get_nns_by_vector(descriptors, 20)
        counts = np.bincount(nearest_idx)
        pred_idx = np.argmax(counts)
        pred = latent_space[pred_idx]
        return pred

    @staticmethod
    def get_descriptors(image):
        return image.flatten()

    def get_characters_from_plate(self, image: np.array) -> List[np.array]:
        bin_image = self.binarize_image(image)
        boxes_list = self.connected_components_boxes(bin_image)
        boxes_list_sorted = self.sort_boxes(boxes_list)
        character_image_list = self.get_cropped_images(image, boxes_list_sorted)
        return character_image_list

    def sort_boxes(self, boxes: List[Tuple]) -> List[np.array]:
        boxes = np.array(boxes)
        return boxes[boxes[:, 0].argsort()]

    @staticmethod
    def binarize_image(image: np.array) -> np.array:
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 131, 15)

    @staticmethod
    def connected_components_boxes(image: np.array) -> List[Tuple[Any, Any, Any, Any]]:
        _, _, boxes, _ = cv2.connectedComponentsWithStats(image)
        # first box is the background
        boxes = boxes[1:]
        y_percentile = np.percentile(boxes[:, 1], 90)
        w_percentile = np.percentile(boxes[:, 2], 10)
        h_percentile = np.percentile(boxes[:, 3], 10)
        pixel_percentile = np.percentile(boxes[:, 4], 10)
        filtered_boxes = []
        for x, y, w, h, pixels in boxes:
            if len(boxes) != 7:
                if pixels > pixel_percentile and h > h_percentile and w > w_percentile and y < y_percentile:
                    filtered_boxes.append((x, y, w, h))
            else:
                filtered_boxes.append((x, y, w, h))
        return filtered_boxes

    def get_cropped_images(self, image: np.array, boxes_list: List[np.array]) -> List[np.array]:
        character_image_list = []
        for box in boxes_list:
            x, y, w, h = box
            char_img = image[y:y+h, x:x+w]
            char_img = self.resize_image(char_img)
            character_image_list.append(char_img)
        return character_image_list

    @staticmethod
    def resize_image(image: np.array, size=(28, 28)) -> np.array:
        return np.array(Image.fromarray(image).resize(size))
