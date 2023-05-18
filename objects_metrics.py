import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


def detect_image_table(filepaths: list[str]) -> pd.DataFrame:
    path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = path_to_tesseract  # Change this path to your Tesseract executable
    data = []
    for filepath in filepaths:
        img = cv2.imread(filepath)
        has_table, empty_fraction = is_table(img)
        has_image = is_image(img)
        data.append([filepath, has_image, has_table, empty_fraction])

    df = pd.DataFrame(data, columns=['filepath', 'hasImage', 'hasTable', 'numberEmpty'])
    return df


def is_empty(image: np.ndarray) -> float:
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    empty = 0
    num_blocks = len(d['level'])
    for i in range(len(d['level'])):
        if d['level'][i] == 2:  # Level 2 corresponds to block-level elements
            print(d['block_num'][i])
            if d['text'][i] == "":
                empty += 1
    return empty / num_blocks


def is_table(image: np.ndarray) -> bool:
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    print(len(d['text']))
    print(len(d['level']))
    has_table = False
    for i in range(len(d['level'])):
        if d['level'][i] == 2:  # Level 2 corresponds to block-level elements
            if d['block_num'][i] != 0:
                if d['block_num'][i] != prev_block_num:
                    if d['conf'][i] < 50:
                        has_table = True
        prev_block_num = d['block_num'][i]
    return has_table


def is_image(image: np.ndarray) -> bool:
    model = MobileNetV2(weights='imagenet', include_top=False)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = tf.expand_dims(image, axis=0)

    features = model.predict(image)
    has_image = features.sum() > 0
    return has_image


def find_non_white_fraction_and_count(
        filepaths: list[str],
        min_area: int = 2500,
        min_width: int = 50,
        min_height: int = 50,
) -> pd.DataFrame:
    data = []
    for image_path in filepaths:

        # Load the image and convert it to grayscale
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to separate non-white elements from the white background
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Find contours of the connected components
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables
        non_white_area = 0
        possible_image_count = 0
        possible_shapes_count = 0
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            # Get the area of the contour
            area = cv2.contourArea(cnt)

            # If the area is larger than the minimum area, consider it as a possible image
            if area > min_area:
                print(rect[1][0], rect[1][1])
                possible_shapes_count += 1
                if rect[1][0] > min_width and rect[1][1] > min_height:
                    possible_image_count += 1

            # Add the area to the total non-white area
            non_white_area += area

        # Calculate the fraction of the image covered by non-white elements
        total_area = image.shape[0] * image.shape[1]
        non_white_fraction = non_white_area / total_area
        data.append([non_white_fraction, possible_shapes_count, possible_image_count])
        print("Fraction of non-white elements:", non_white_fraction)
        print("Possible shapes count:", possible_shapes_count)
        print("Possible image count:", possible_image_count)
    df = pd.DataFrame(data, columns=['fraction', 'possible_shapes', 'possible_images'])
    return df
