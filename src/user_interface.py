import os
from pathlib import Path
from typing import List, Text

import numpy as np

from utils import show_image, show_split_chars, show_contour_detection, write_image


class Menu:

    def __init__(self) -> None:
        # Files
        self.file_list = None
        self.current_file = None
        # Paths
        self.localization_path = Path("../images/localization")
        self.segmentation_path = Path("../images/segmentation")
        self.ocr_path = Path("../images/ocr")
        self.predictions_path = Path("../images/predictions")
        # Mode
        self.show = True
        self.write = False

    def current_file(self, file):
        self.current_file = file

    def on_init(self):
        txt = f"{'#' * 50} MENU {'#' * 50}\n"
        txt += f"# WELCOME! \n"
        txt += "# This is an interactive UI. Please start by choosing in which mode you want to run the script:\n"
        print(txt)
        mode = input("# Available modes: 'predict', 'visualize'\n"
                     "# --> 'predict': no user interaction, only get predictions.\n"
                     "# --> 'visualize': visualize each step of the algorithm.\n").lower()
        self.load_mode(mode)
        write = input("# Would you like to wirte the generated images following the project governance? (y/n)\n").lower()
        self.load_write_mode(write)
        path = Path(input("# Please enter the path with the image or folder containing the images to process:\n"))
        self.load_paths(path)
        txt = "The following images are going to get processed:\n"
        txt += f"{[f.name for f in self.file_list]}\n"
        print(txt)

    def load_mode(self, mode: Text):
        if mode == 'predict':
            self.show = False
        elif mode == 'visualize':
            self.show = True
        else:
            raise NotImplementedError("Option not supported. Please choose between 'predict' or 'visualize'.")

    def load_write_mode(self, write):
        if write in ['y', 'yes']:
            self.write = True

    def load_paths(self, path: Path):
        if os.path.isdir(path):
            self.file_list = [Path(os.path.abspath(path / f)) for f in os.listdir(path)]
        elif os.path.isfile(path):
            self.file_list = [Path(os.path.abspath(path))]
        else:
            raise FileNotFoundError("Given file/path was not found.")

    def on_end(self):
        if self.current_file != self.file_list[-1].name:
            if self.show:
                go = input("# Would you like to continue? (y/n)\n").lower()
                if go in ["y", "yes"]:
                    print("# Next image...")
                    print()
                else:
                    self.finish()
        else:
            self.finish()

    @staticmethod
    def finish():
        print("# All images were processed.")
        print("# Thanks for using this project!")
        print('#' * 105)
        quit()

    def show_new_file(self):
        print(f"# New image --> {self.current_file}")
        print("# Loading and resizing...")
        print()

    def display_original_image(self, image: np.array) -> None:
        if self.show:
            go = input("# Would you like to display the original image? (y/n)\n").lower()
            print()
            if go in ["y", "yes"]:
                show_image(image, "Original image")

    def display_contour_detection(self, image: np.array, edged_image: np.array, contours: List) -> None:
        if self.show:
            go = input("# Would you like to display the contour detection? (y/n)\n").lower()
            print()
            if go in ["y", "yes"]:
                show_contour_detection(image, edged_image, contours)

    def display_projected_image(self, image: np.array) -> None:
        if self.show:
            go = input("# Would you like to display the projected image? (y/n)\n").lower()
            print()
            if go in ["y", "yes"]:
                show_image(image, "Projected image")
        if self.write:
            path = self.localization_path / f"{self.current_file[:-4]}_localization{self.current_file[-4:]}"
            print(f"# Image wrote at {path}")
            print()
            write_image(path, image)

    def display_segmented_image(self, image: np.array) -> None:
        if self.show:
            go = input("# Would you like to display the segmented image? (y/n)\n").lower()
            print()
            if go in ["y", "yes"]:
                show_image(image, "Segmented image")
        if self.write:
            path = self.segmentation_path / f"{self.current_file[:-4]}_segmentation{self.current_file[-4:]}"
            print(f"# Image wrote at {path}")
            print()
            write_image(path, image)

    def display_split_chars(self, chars_list: List[np.array]) -> None:
        if self.show:
            go = input("# Would you like to display the characters split? (y/n)\n").lower()
            print()
            if go in ["y", "yes"]:
                show_split_chars(chars_list, "Split text")
        if self.write:
            path = self.ocr_path / f"{self.current_file[:-4]}"
            chars_path = path / "chars"
            digits_path = path / "digits"
            chars_path.mkdir(parents=True, exist_ok=True)
            digits_path.mkdir(parents=True, exist_ok=True)
            print(f"# All characters were wrote at {path}/")
            print()
            for idx, image in enumerate(chars_list):
                if idx < 4:
                    file_path = digits_path / f"{idx}{self.current_file[-4:]}"
                else:
                    file_path = chars_path / f"{idx}{self.current_file[-4:]}"
                write_image(file_path, image)

    def display_predicted_plate(self, image: np.array, prediction: Text) -> None:
        print(f"# PREDICTED PLATE: {prediction}")
        print()
        if self.write:
            path = self.predictions_path / f"{self.current_file[:-4]}_prediction_{prediction}{self.current_file[-4:]}"
            write_image(path, image)

    def detection_failed(self):
        print(f"# Detection for file {self.current_file} failed.")
        go = input("# Would you like to continue? (y/n)\n").lower()
        print()
        if go in ["n", "no"]:
            quit()
