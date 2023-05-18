import cv2
import pytesseract
from pytesseract import Output
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


def detect_image_table(filepaths):
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


def is_empty(image):
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    empty = 0
    num_blocks = len(d['level'])
    for i in range(len(d['level'])):
        if d['level'][i] == 2:  # Level 2 corresponds to block-level elements
            print(d['block_num'][i])
            if d['text'][i] == "":
                empty += 1
    return empty / num_blocks


def is_table(image):
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


def is_image(image):
    model = MobileNetV2(weights='imagenet', include_top=False)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = tf.expand_dims(image, axis=0)

    features = model.predict(image)
    has_image = features.sum() > 0
    return has_image

