import argparse
import spacy
import pandas as pd
from pathlib import Path
from brand_detector.predict import predict_df, predict_df_score, preds2corrects

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained spaCy model for brand detection."
    )
    parser.add_argument("model_path", type=str, help="trained model directory")
    parser.add_argument("test_data_path", type=str, help="path to test_data.json")
    parser.add_argument(
        "-s",
        "--score",
        help="return confidence score for each prediction (uses beam search)",
        action="store_true",
    )
    args = parser.parse_args()

    test_data_path = Path(test_data_path)
    df_test = pd.read_json(test_data_path)
    nlp = spacy.load(args.model_path)
    if args.score:
        df_test = predict_df_score(nlp, df_test)
        df_test.to_json(test_data_path.parent / "test_data_pred_s.json")
    else:
        df_test = predict_df(nlp, df_test)
        df_test["corrects"], val_acc = preds2corrects(
            df_test["predictions"], df_test["brand"]
        )
        df_test.to_json(test_data_path.parent / "test_data_pred.json")
        print(val_acc)
