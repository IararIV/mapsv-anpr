from numbers import Number
from typing import Optional, List

import numpy as np

import cv2


def matrix_to_corners(matrix: np.array) -> List[List[Number]]:
    corners = []
    for i in range(0, 4):
        corners.append([matrix[i][0][0], matrix[i][0][1]])

    corners = np.array(corners)
    ordered = corners[corners[:, 0].argsort()]

    left_corners = ordered[:2]
    right_corners = ordered[2:]

    top_left = left_corners[left_corners[:, 1].argsort()][0]
    bottom_left = left_corners[left_corners[:, 1].argsort()][1]

    top_right = right_corners[right_corners[:, 1].argsort()][0]
    bottom_right = right_corners[right_corners[:, 1].argsort()][1]

    return [list(top_left), list(top_right), list(bottom_left), list(bottom_right)]


class DetectPlateContour:

    def __init__(self, width: Number = 1100, epsilon: Number = 2):
        self.width = width
        self.epsilon = epsilon
        # Steps for visualization
        self.edged = None
        self.cv2_contours = []

    def detect(self, image: np.array) -> Optional[List[List[Number]]]:

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)

        self.edged = cv2.Canny(gray_image, 30, 200)

        contours, new = cv2.findContours(self.edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.cv2_contours.append(contours)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        self.cv2_contours.append(contours)

        screen_cnt = None
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
            x, y, w, h = cv2.boundingRect(c)
            if len(approx) == 4 and abs((w/h) - 340/110) < self.epsilon:
                screen_cnt = approx
                break

        if isinstance(screen_cnt, (np.ndarray, np.generic)):
            self.cv2_contours.append([screen_cnt])
            corners = matrix_to_corners(screen_cnt)
            return corners
        else:
            return None
