from typing import Tuple

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm.notebook import tqdm 


def is_empty(image: np.ndarray) -> float:
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    empty = 0
    num_blocks = len(d['level'])
    for i in range(len(d['level'])):
        if d['level'][i] == 2:  # Level 2 corresponds to block-level elements
            if d['text'][i] == "":
                empty += 1
    return empty / num_blocks


def is_empty_plain(image: np.ndarray) -> float:
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    empty = 0
    num_blocks = len(d['text'])
    for i in range(len(d['level'])):
        if d['text'][i] == "":
            empty += 1
    return empty / num_blocks


def is_table(image: np.ndarray) -> bool:
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    has_table = False
    for i in range(len(d['level'])):
        if d['level'][i] == 2:  # Level 2 corresponds to block-level elements
            if d['block_num'][i] != 0:
                if d['block_num'][i] != prev_block_num:
                    if d['conf'][i] < 50:
                        has_table = True
        prev_block_num = d['block_num'][i]
    return has_table


# def is_image(image: np.ndarray) -> bool:
#     model = MobileNetV2(weights='imagenet', include_top=False)
#     image = cv2.resize(image, (224, 224))
#     image = img_to_array(image)
#     image = preprocess_input(image)
#     image = tf.expand_dims(image, axis=0)
#
#     features = model.predict(image)
#     has_image = features.sum() > 0
#     return has_image


def generate_metrics(
        filepaths: list[str],
        min_area: int = 2500,
        min_width: int = 50,
        min_height: int = 50,
        section_size: Tuple[float, float] = (0.5, 0.5),
) -> pd.DataFrame:
    path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = path_to_tesseract  # Change this path to your Tesseract executable
    data = []
    for filepath in tqdm(filepaths):
        print(filepath)

        img = cv2.imread(filepath)
        has_table = is_table(img)
        empty_fraction = is_empty(img)
        fraction, possible_shapes, possible_images = find_shapes(img, min_area, min_width, min_height)
        # fraction_sections = mask(img, section_size)
        data.append([filepath, has_table, empty_fraction, fraction, possible_shapes, possible_images])

    df = pd.DataFrame(data, columns=['filepath', 'hasTable', 'numberEmpty', 'nonWhiteFraction', 'possibleShapes', 'possibleImages'])
    return df


def find_shapes(
        image: np.ndarray,
        min_area: int = 2500,
        min_width: int = 50,
        min_height: int = 50,
) -> Tuple[float, int, int]:
    # Load the image and convert it to grayscale
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
            possible_shapes_count += 1
            if rect[1][0] > min_width and rect[1][1] > min_height:
                possible_image_count += 1

        # Add the area to the total non-white area
        non_white_area += area

    # Calculate the fraction of the image covered by non-white elements
    total_area = image.shape[0] * image.shape[1]
    non_white_fraction = non_white_area / total_area
    return non_white_fraction, possible_shapes_count, possible_image_count


# def find_non_white_fraction_and_count(
#         filepaths: list[str],
#         min_area: int = 2500,
#         min_width: int = 50,
#         min_height: int = 50,
# ) -> pd.DataFrame:
#     data = []
#     for image_path in filepaths:
#         img = cv2.imread(image_path)
#         data.append(find_shapes(img, min_area, min_width, min_height))
#     df = pd.DataFrame(data, columns=['filepath', 'fraction', 'possible_shapes', 'possible_images'])
#     return df


def mask(img: np.ndarray, section_size: Tuple[float, float] = (0.5, 0.5), simplify=True) -> float:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape
    section_height = int(img_height * section_size[0])
    section_width = int(img_width * section_size[1])

    # Initialize counters
    total_sections = 0
    non_empty_sections = 0
    # Iterate through sections
    for y in range(0, img_height, section_height):
        for x in range(0, img_width, section_width):
            total_sections += 1

            # Create a mask to cover all sections except the current one
            mask = img.copy()
            mask[0:img_height, 0:img_width] = 255
            mask[y:y + section_height, x:x + section_width] = img[y:y + section_height, x:x + section_width]
            empty = 1.0
            # Perform OCR with pytesseract
            if not simplify:
                empty = is_empty_plain(mask)
            else:
                # Perform OCR with pytesseract
                text = pytesseract.image_to_string(mask)

                # If there's any text, consider the section non-empty
                if text.strip():
                    non_empty_sections += 1

            # If there's any text, consider the section non-empty
            # print(empty, "\n------")
            if empty < 0.95:
                non_empty_sections += 1

    # Calculate the fraction of non-empty sections
    return non_empty_sections / total_sections


# def non_empty_sections_fraction(filepaths: list[str], section_size: Tuple[float, float] = (0.5, 0.5)) -> pd.DataFrame:
#     data = []
#     for image_path in filepaths:
#         # Read image
#         img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         fraction = mask(img, section_size)
#         # Calculate section dimensions
#
#         data.append([image_path, fraction])
#     df = pd.DataFrame(data, columns=['filepath', 'fraction_non_empty'])
#     return df


# def train_and_concat_all(filepaths: list[str]) -> pd.DataFrame:
#     detect_image_table_empty: pd.DataFrame = detect_image_table(filepaths)
#     non_white_shapes_and_images: pd.DataFrame = find_non_white_fraction_and_count(filepaths)
#     # non_empty_sections: pd.DataFrame = non_empty_sections_fraction(filepaths)
#
#     return pd.merge(detect_image_table_empty, non_white_shapes_and_images,
#                     on='filepath')  # , non_empty_sections, on='filepath')


def get_filenames(folder_path):
    filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".tiff"):
                filenames.append(os.path.join(root, file))
    return filenames

