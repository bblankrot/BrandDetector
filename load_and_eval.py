import argparse
import spacy
import pandas as pd

from brand_detector.predict import predict, predict_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained spaCy model for brand detection."
    )
    parser.add_argument("dir", type=str, help="trained model directory")
    parser.add_argument(
        "-s",
        "--score",
        help="return confidence score for each prediction (uses beam search)",
        action="store_true",
    )
    args = parser.parse_args()
    df_test = pd.read_json("data/preprocessed/test_data.json")
    nlp = spacy.load(args.dir)
    if args.score:
        df_test = predict_score(nlp, df_test)
        df_test.to_json("data/preprocessed/test_data_pred_s.json")
    else:
        df_test = predict(nlp, df_test)
        df_test.to_json("data/preprocessed/test_data_pred.json")
    

