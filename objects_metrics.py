from typing import Tuple

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import pandas as pd
import os


def is_empty(image: np.ndarray) -> float:
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    empty = 0
    num_blocks = len(d['level'])
    for i in range(len(d['level'])):
        if d['level'][i] == 2:  # level 2 means block-level elements
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
        if d['level'][i] == 2:
            if d['block_num'][i] != 0:
                if d['block_num'][i] != prev_block_num:
                    if d['conf'][i] < 50:
                        has_table = True
        prev_block_num = d['block_num'][i]
    return has_table


def generate_metrics(
        filepaths: list[str],
        min_area: int = 2500,
        min_width: int = 50,
        min_height: int = 50,
        section_size: Tuple[float, float] = (0.5, 0.5),
) -> pd.DataFrame:
    """
    This function processes a list of image filepaths, analyzes each image for various characteristics like
    emptiness, shape detection, and section masking, and returns a pandas DataFrame with the results of the analysis.
    Parameters
    ----------
    filepaths: (list[str]):
    A list of filepaths to the images that need to be processed.

    min_area: (int, optional):
    The minimum area for a shape to be considered. Defaults to 2500.

    min_width: (int, optional):
    The minimum width for a shape to be considered. Defaults to 50.

    min_height: (int, optional):
    The minimum height for a shape to be considered. Defaults to 50.

    section_size: (Tuple[float, float], optional):
    A tuple of two float values representing the relative width and
    height of the sections to be used for section masking. Defaults to (0.5, 0.5).

Returns:

    'filepath': The filepath of the image

    'numberEmpty': The fraction of empty blocks in the image

    'nonWhiteFraction': The fraction of non-white pixels in the image

    'possibleShapes': The number of possible shapes detected in the image

    'possibleImages': The number of possible images detected in the image

    'nonEmptySections': The number of non-empty sections in the image after applying section masking
    """
    path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = path_to_tesseract
    data = []
    for filepath in filepaths:
        img = cv2.imread(filepath)
        empty_fraction = is_empty(img)
        fraction, possible_shapes, possible_images = find_shapes(img, min_area, min_width, min_height)
        fraction_sections = mask(img, section_size)
        data.append([filepath, empty_fraction, fraction, possible_shapes, possible_images, fraction_sections])

    df = pd.DataFrame(data, columns=['filepath', 'numberEmpty', 'nonWhiteFraction', 'possibleShapes',
                                     'possibleImages', 'nonEmptySections'])
    return df


def find_shapes(
        image: np.ndarray,
        min_area: int = 2500,
        min_width: int = 50,
        min_height: int = 50,
) -> Tuple[float, int, int]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    non_white_area = 0
    possible_image_count = 0
    possible_shapes_count = 0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        area = cv2.contourArea(cnt)
        if area > min_area:
            possible_shapes_count += 1
            if rect[1][0] > min_width and rect[1][1] > min_height:
                possible_image_count += 1
        non_white_area += area
    total_area = image.shape[0] * image.shape[1]
    non_white_fraction = non_white_area / total_area
    return non_white_fraction, possible_shapes_count, possible_image_count


def mask(img: np.ndarray, section_size: Tuple[float, float] = (0.5, 0.5)) -> float:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape
    section_height = int(img_height * section_size[0])
    section_width = int(img_width * section_size[1])
    total_sections = 0
    non_empty_sections = 0
    for y in range(0, img_height, section_height):
        for x in range(0, img_width, section_width):
            total_sections += 1
            mask = img.copy()
            mask[0:img_height, 0:img_width] = 255
            mask[y:y + section_height, x:x + section_width] = img[y:y + section_height, x:x + section_width]
            empty = is_empty_plain(mask)
            if empty < 0.95:
                non_empty_sections += 1
    return non_empty_sections / total_sections


def get_filenames(folder_path):
    filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".tiff"):
                filenames.append(os.path.join(root, file))
    return filenames

