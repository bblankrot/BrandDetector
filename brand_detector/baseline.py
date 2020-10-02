import pandas as pd
import numpy as np
import statistics
from .utils import preprocess, find_brands, generate_train_test_set
from .predict import equal_brands


def baseline_accuracy(brands_list, df):
    result = df.apply(
        lambda row: baseline_accuracy_row(brands_list, row),
        axis=1,
        result_type="expand",
    )

    acc = result["bl_corrects"].sum() / result.shape[0]
    return acc, result


def baseline_accuracy_row(brands_list, row):
    preds = []
    for brand in brands_list:
        matches = find_brands(brand, row["transcription"])
        for _ in matches:
            preds.append(brand)
    # now find most common value(s)
    preds = statistics.multimode(preds)
    correct = any(equal_brands(pred, row["brand"]) for pred in preds)
    # correct = any(row["brand"].lower() == pred.lower() for pred in preds)
    return pd.Series([preds, correct], index=["bl_preds", "bl_corrects"])
