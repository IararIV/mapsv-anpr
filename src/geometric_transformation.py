from numbers import Number
from typing import List

import numpy as np

import cv2


class GeometricProjection:

    def __init__(self):
        self.new_coords = np.float32([[0, 0], [340, 0], [0, 110], [340, 110]])
        self.output_size = (340, 110)

    def project(self, image: np.array, corners: List[List[Number]]) -> np.array:
        ###
        # corners: (xx, xy, yx, yy)
        #   tl: top left (1)
        #   tr: top right (2)
        #   bl: bottom left (3)
        #   br: bottom right (4)
        ###

        tl, tr, bl, br = corners

        source = np.float32([tl, tr, bl, br])

        # getPerspectiveTransform: 3x3 transform matrix generated from two sets of coordinates
        mat = cv2.getPerspectiveTransform(source, self.new_coords)

        # warpPerspective: apply transformation using the computed conversion matrix
        img_dst = cv2.warpPerspective(image, mat, self.output_size)

        return img_dst
