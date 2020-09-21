import argparse
import spacy
import pandas as pd

def load_test_set(filename, pct):
    df = pd.read_json(filename)
    n_rows = df.shape[0]
    # TODO: shuffle selection
    n_rows = int(round(n_rows * pct))
    return df.iloc[:n_rows, :]


def predict(model_dir, df):
    nlp = spacy.load(model_dir)
    preds = []
    for _, row in df.iterrows():
        doc = nlp(row["transcription"])
        pred = []
        for ent in doc.ents:
            if ent.label_ == "BRAND":
                pred.append(ent.text)
            # pred.append((ent.label_, ent.text))
        preds.append(pred)
    df["predictions"] = preds
    return df


def predict_score(model_dir, df, beam_width=16, beam_density=0.0001, threshold=0.2):
    nlp = spacy.load(model_dir)
    preds = []
    for _, row in df.iterrows():
        with nlp.disable_pipes("ner"):
            doc = nlp(row["transcription"])
        beams = nlp.entity.beam_parse(
            [doc], beam_width=beam_width, beam_density=beam_density
        )
        pred = []
        for beam in beams:
            for score, ents in nlp.entity.moves.get_beam_parses(beam):
                for start, end, label in ents:
                    if label == "BRAND" and score > threshold:
                        match = doc[start:end].text
                        pred.append([match, score])
        preds.append(pred)
    df["predictions"] = preds
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained spaCy model for brand detection."
    )
    parser.add_argument("dir", type=str, help="trained model directory")
    parser.add_argument(
        "-p",
        "--pct",
        type=float,
        help="fraction of testing data to look at",
        default=0.01,
    )
    parser.add_argument(
        "-s",
        "--score",
        help="return confidence score for each prediction (uses beam search)",
        action="store_true",
    )
    args = parser.parse_args()
    df_test = load_test_set("../data/preprocessed/test_data.json", args.pct)
    if args.score:
        df_test = predict_score(args.dir, df_test)
    else:
        df_test = predict(args.dir, df_test)
    df_test.to_json("../data/preprocessed/test_data_pred.json")

