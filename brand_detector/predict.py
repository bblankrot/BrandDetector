import re
import statistics
import warnings
import pandas as pd


def predict(nlp, transcriptions, brands_list=None):
    flag_str = type(transcriptions) == str
    if flag_str:
        transcriptions = [transcriptions]

    if brands_list is not None:
        df_brands = pd.read_csv(brands_list)

    preds = []
    for transcript in transcriptions:
        doc = nlp(transcript)
        pred = []
        for ent in doc.ents:
            if ent.label_ == "BRAND":
                brand = ent.text
                if brands_list is not None:
                    industry = brand2industry(brand, df_brands)
                    pred.append((brand, industry))
                else:
                    pred.append(brand)
        preds.append(pred)

    if flag_str:
        return preds[0]
    else:
        return preds


def brand2industry(brand, df_brands):
    industries = df_brands.loc[df_brands["brand"] == brand, "industry"]
    if industries.shape[0] == 0:
        return ""
    industry = industries.iloc[0]
    return "" if pd.isna(industry) else industry


def predict_df(nlp, df):
    """Predict the entities (brands) in each transcription of a DataFrame."""
    preds = []
    for _, row in df.iterrows():
        doc = nlp(row["transcription"])
        pred = []
        for ent in doc.ents:
            if ent.label_ == "BRAND":
                pred.append(ent.text)
        preds.append(pred)
    df["predictions"] = preds
    return df


def predict_df_score(nlp, df, beam_width=16, beam_density=0.0001, threshold=0.2):
    """Predict entities using beam search, so it returns probabilities as well."""
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


def equal_brands(prediction, brand):
    """Calculate if two brands (correct and predicted) are equal, up to some allowed
    deviations."""

    def inner_text(text):
        inner = text.replace("-", " ").strip().lower()
        inner = re.sub(r"\W*$", "", re.sub(r"^\W*", "", inner))
        if inner == "":
            warnings.warn(
                "Encountered bad prediction/brand: {} / {}".format(prediction, brand),
                RuntimeWarning,
            )
            return None
        return inner

    prediction_d = inner_text(prediction)
    brand_d = inner_text(brand)
    return prediction_d == brand_d


def preds2corrects(predictions, brands):
    """Check if correct prediction is one of most common, saving it in another column."""
    multimodes = predictions.apply(
        lambda a: [pred.lower().replace("-", " ") for pred in a]
    )
    multimodes = multimodes.apply(statistics.multimode)
    acc = 0
    corrects = []
    for i, item in multimodes.iteritems():
        correct = False
        if not len(item):
            corrects.append(correct)
            continue
        for it in item:
            if equal_brands(it, brands[i]):
                acc += 1
                correct = True
                break
        corrects.append(correct)
    return corrects, acc / multimodes.shape[0]


def f1score(predictions, corrects):
    """Return F1 score of predictions, using micro averaging.
    Equal to 2 * accuracy / (accuracy + 1)"""
    tp = corrects.sum()
    fp = (~corrects[predictions.apply(len) > 0]).sum()
    fn = (
        ~corrects[predictions.apply(len) == 0]
    ).sum()  # or N - (tp + fp), since tn = 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1
