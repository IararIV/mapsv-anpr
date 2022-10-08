from localization import DetectPlateContour
from geometric_transformation import GeometricProjection
from ocr import OCR

from segmentation import ImageSegmentation
from user_interface import Menu
from utils import load_image, resize_image


def main():

    menu = Menu()
    menu.on_init()
    file_list = menu.file_list

    detect_plate = DetectPlateContour()
    geometric_projection = GeometricProjection()
    segment_plate = ImageSegmentation()
    ocr = OCR()

    for image_file in file_list:
        # Load image and resize
        menu.current_file = image_file.name
        menu.show_new_file()
        img = load_image(image_file)
        img = resize_image(img)
        menu.display_original_image(img)
        # Detect corners
        corners = detect_plate.detect(img)
        menu.display_contour_detection(img, detect_plate.edged, detect_plate.cv2_contours)
        if not corners:
            menu.detection_failed()
            continue
        # project image
        img_proj = geometric_projection.project(img, corners)
        menu.display_projected_image(img_proj)
        # segment image
        img_sgm = segment_plate.segment(img_proj)
        menu.display_segmented_image(img_sgm)
        # char split
        char_list = ocr.get_characters_from_plate(img_sgm)
        menu.display_split_chars(char_list)
        # prediction
        plate_prediction = ""
        for idx, char in enumerate(char_list):
            prediction = ocr.predict_char(idx, char)
            plate_prediction += prediction
        menu.display_predicted_plate(img_proj, plate_prediction)
        menu.on_end()


if __name__ == '__main__':
    main()
