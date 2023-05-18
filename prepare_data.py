import pandas as pd

def read_files():

    id2label = pd.read_pickle('data/id2label_final.pkl')
    label2id = pd.read_pickle('data/label2id_final.pkl')
    submission_file = pd.read_csv('data/submission_file.csv')
    test_ids_final = pd.read_csv('data/test_ids_final.csv')
    test_ocr_clean = pd.read_pickle('data/test_ocr_clean.pkl')
    train_ids_final = pd.read_csv('data/train_ids_final.csv')
    train_labels_final = pd.read_pickle('data/train_labels_final.pkl')
    train_set_ocr = pd.read_pickle('data/train_set_ocr.pkl')

    return id2label, label2id, submission_file, test_ids_final, test_ocr_clean, train_ids_final, train_labels_final, train_set_ocr

# id2label, label2id, submission_file, test_ids_final, test_ocr_clean, train_ids_final, train_labels_final, train_set_ocr = read_files()

# Goal: dict = {'file_name.jpg':('text',label_id)}

def transform_image_names(image_label_dict, sep='\\'):

    new_dict = {key.split(sep)[-1]: value for key, value in image_label_dict.items()}

    return new_dict

# train_image_text_dict = transform_image_names(train_set_ocr)
# train_image_label_dict = transform_image_names(train_labels_final)
# test_image_text_dict = transform_image_names(test_ocr_clean, sep='/')

def create_final_dict(image_text_dict, image_label_dict, id2label):

    result_dict = {}

    for file_name in image_text_dict:
        if file_name in image_label_dict:
            text = image_text_dict[file_name]
            label = image_label_dict[file_name]
            label_id = id2label[label]
            result_dict[file_name] = (text, label_id)

    return result_dict

# train_final_dict = create_final_dict(train_image_text_dict, train_image_label_dict, id2label)