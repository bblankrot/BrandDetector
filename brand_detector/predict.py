import re
import statistics


def predict(nlp, df):
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


def predict_score(nlp, df, beam_width=16, beam_density=0.0001, threshold=0.2):
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
    def inner_text(text):
        inner = re.search(r"(\w.*\w)", text.replace("-", " ").strip().lower())
        if inner is None:
            raise ValueError("Encountered bad brand name")
        return inner[0]

    prediction_d = inner_text(prediction)
    brand_d = inner_text(brand)
    return prediction_d == brand_d


def preds2corrects(predictions, brands):
    multimodes = predictions.apply(statistics.multimode)
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
