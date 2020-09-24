import pandas as pd
import numpy as np

# from collections import defaultdict
import re
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import argparse
import json
import time
from brand_detector.load_and_eval import predict

def preprocess(df):
    df["brand"] = df["brand"].str.replace(r"-tier[1-3]$", "", regex=True)
    df.loc[df["brand"] == "Zico", "brand"] = "Geico"
    df.loc[df["brand"].str.lower() == "the home depot", "brand"] = "Home Depot"
    df.loc[df["brand"] == "t", "brand"] = "T-Mobile"
    return df


def find_brands(brand, text, ignore_case=True, dehyphenate=True):
    # TODO: handle edge cases such as: LV (Liverpool Victoria),
    # "ashley home store" <-> "ashley homestore"
    # "Macy's" in brand but "Macys" in string
    # "U.S. Waterproofing" <-> "US Waterproofing"
    if brand == "":
        return []
    brand_d = brand.replace("-", " ") if dehyphenate else brand
    text_d = text.replace("-", " ") if dehyphenate else text
    # if there's punctuation at the beginning or end of the brand, remove it
    inner_brand = re.match(r"([a-zA-Z0-9()].*[a-zA-Z0-9()])", brand_d)
    if inner_brand is None:
        if len(brand_d) > 1:
            raise RuntimeError("encountered problem for brand: {}".format(brand))
    else:
        brand_d = inner_brand[0]
    brand_d = "\\b" + re.escape(brand_d) + "\\b"
    if ignore_case:
        try:
            found_iters = re.finditer(brand_d, text_d, re.IGNORECASE)
        except:
            raise RuntimeError(
                "encountered problem for brand: {} ({})".format(brand, brand_d)
            )
    else:
        found_iters = re.finditer(brand_d, text_d)
    matches = []
    for match in found_iters:
        s, e = match.start(), match.end()
        matches.append((s, e))
    return matches


def find_brands_in_df(df):
    df2 = df.copy()
    df2["brand_matches"] = df2.apply(
        lambda row: find_brands(row["brand"], row["transcription"]), axis=1
    )
    return df2


def df_to_entity_list(df):
    train = []
    for _, row in df.iterrows():
        if not row["brand_matches"]:
            continue  # should not happen
        matches = []
        for match in row["brand_matches"]:
            matches.append((match[0], match[1], "BRAND"))
        train.append((row["transcription"], {"entities": matches}))
    return train


def generate_train_test_set(filename, pct):
    df = pd.read_json(filename)
    df = find_brands_in_df(df)
    df_train_test = df[df["brand_matches"].str.len() > 0]
    num_rows = df_train_test.shape[0]
    num_train = int(round(num_rows * pct))
    df_train, df_test = df.iloc[:num_train, :].copy(), df.iloc[num_train:, :].copy()
    df_dirty = df[df["brand_matches"].str.len() == 0].copy()

    entity_list = df_to_entity_list(df_train)

    seen_set = set(df_train["brand"].to_numpy())
    df_test["seen_in_training"] = df_test["brand"].apply(lambda x: x in seen_set)
    # not strictly necessary, since dirty dataset does not have brand name in transcript, but might be interesting
    df_dirty["seen_in_training"] = df_dirty["brand"].apply(lambda x: x in seen_set)
    return entity_list, df_test, df_dirty


def calculate_accuracy(nlp, val):

    return 0.0


def train_spacy(
    entity_list,
    model=None,
    new_label="BRAND",
    new_model_name="brand",
    output_dir=str(Path.home() / "models"),
    n_iter=30,
    val=None,
):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.load("en_core_web_sm")  # load base spaCy model
        print("Loaded model '%s'" % model)
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(new_label)  # add new entity label to entity recognizer

    optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module="spacy")

        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        start_time = time.time()
        history = []
        for i in range(n_iter):
            losses = {}
            val_acc = {}
            random.shuffle(entity_list)
            batches = minibatch(entity_list, size=sizes)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            cur_time = time.time()
            print("Losses", losses)
            print("({:.2f} seconds)".format(cur_time - start_time))
            start_time = cur_time

            # save to disk after each iteration
            epoch_path = output_dir / "{0}_epoch_{1}".format(new_model_name, i)
            nlp.to_disk(epoch_path)

            if val is not None:
                val_acc = calculate_accuracy(nlp, val)
            history.append({"losses": losses, "val_accuracy": val_acc})

    # test the trained model
    test_text = (
        "Did you know switching to Geico could save you 15 percent "
        "or more on car insurance?"
    )
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    nlp.meta["name"] = new_model_name  # rename model
    nlp.to_disk(output_dir / new_model_name)
    print("Saved model to", output_dir)

    # test the saved model
    print("Loading from", output_dir)
    nlp2 = spacy.load(output_dir / new_model_name)
    # Check the classes have loaded back consistently
    assert nlp2.get_pipe("ner").move_names == move_names
    doc2 = nlp2(test_text)
    for ent in doc2.ents:
        print(ent.label_, ent.text)
    with open(output_dir / new_model_name / "history.json", "w") as fp:
        json.dump(history, fp)

    return nlp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train spaCy model for brand detection."
    )
    parser.add_argument(
        "-m", "--model", type=str, help="base model name", default="en_core_web_sm"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="output model directory",
        default=str("models"),
    )
    parser.add_argument(
        "-n", "--name", type=str, help="output model name", default="brand_sm"
    )
    parser.add_argument(
        "-i", "--iters", type=int, help="output model directory", default=10
    )
    parser.add_argument(
        "-p",
        "--pct",
        type=float,
        help="fraction of training data from entire labeled dataset",
        default=0.01,
    )
    parser.add_argument(
        "--gpu",
        help=(
            "require GPU (it is preferred either way, "
            "but this raises an error if a GPU is unavailable)"
        ),
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--validate",
        help=("calculate accuracy on test/val set during training"),
        action="store_true",
    )
    args = parser.parse_args()

    if args.gpu:
        spacy.require_gpu()
    else:
        has_gpu = spacy.prefer_gpu()
        print("Using GPU" if has_gpu else "No GPU, training on CPU")

    entity_list, df_test, df_dirty = generate_train_test_set(
        "../data/raw/listen_demo_records.json", args.pct
    )
    df_test.to_json("../data/preprocessed/test_data.json")
    df_dirty.to_json("../data/preprocessed/dirty_data.json")
    train_spacy(
        entity_list,
        model=args.model,
        new_model_name=args.name,
        output_dir=args.output,
        n_iter=args.iters,
        val=(df_test if args.validate else None),
    )
