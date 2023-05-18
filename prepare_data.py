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
