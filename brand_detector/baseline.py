import pandas as pd
import numpy as np
import statistics
from .utils import preprocess, find_brands, generate_train_test_set


def get_baseline_accuracy(brands_list, df):
    acc = 0
    for _, row in df.iterrows():
        preds = []
        for brand in brands_list:
            matches = find_brands(brand, row["transcription"])
            for _ in matches:
                preds.append(brand)
        # now find most common value(s)
        preds = statistics.multimode(preds)
        if row["brand"] in preds:
            acc += 1
    acc /= df.shape[0]
    return acc


def baseline_accuracy(brands_list, df):
    result = df.apply(lambda row: baseline_accuracy_row(brands_list, row), axis=1, result_type='expand')

    acc = result['corrects'].sum() / result.shape[0]
    return acc, result


def baseline_accuracy_row(brands_list, row):
    preds = []
    for brand in brands_list:
        matches = find_brands(brand, row["transcription"])
        for _ in matches:
            preds.append(brand)
    # now find most common value(s)
    preds = statistics.multimode(preds)
    return pd.Series([row["brand"] in preds, preds], index=['corrects', 'preds'])


# import brand_detector
# import numpy as np
# df_train, df_test, df_dirty = brand_detector.utils.generate_train_test_set('data/raw/listen_demo_records.json', 0.8)
# train_brands = np.unique(df_train["brand"].to_numpy())
# acc, result = brand_detector.baseline.baseline_accuracy(train_brands, df_test)
# result.to_json('baseline_corrects.json')
