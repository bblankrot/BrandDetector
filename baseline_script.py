import brand_detector
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate baseline accuracy for brand detection."
    )
    parser.add_argument(
        "-a",
        "--apostrophe",
        help=("remove apostrophes from brand and transcription"),
        action="store_true",
    )
    parser.add_argument("--data", help=("add additional data"), default=None)
    args = parser.parse_args()
    df_train, df_test, df_dirty = brand_detector.utils.generate_train_test_set(
        "data/raw/listen_demo_records.json",
        0.8,
        args.apostrophe,
        additional_data=args.data,
    )
    train_brands = np.unique(df_train["brand"].to_numpy())
    acc, result = brand_detector.baseline.baseline_accuracy(train_brands, df_test)
    result.to_json("baseline_corrects.json")
    print(acc)

