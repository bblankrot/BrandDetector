import pandas as pd
import numpy as np
import re
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import json
import time
import statistics

from .utils import find_brands
from .predict import predict, preds2corrects


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


def calculate_accuracy(df):
    multimode = df["predictions"].apply(statistics.multimode)
    acc = 0
    for i, item in multimode.iteritems():
        if not len(item):
            continue
        for it in item:
            if find_brands(df["brand"][i], it):
                acc += 1
                break
    return acc / df.shape[0]


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
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(new_label)  # add new entity label to entity recognizer

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module="spacy")

        sizes = compounding(4.0, 32.0, 1.001)
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
                preds = predict(nlp, val)
                _, val_acc = preds2corrects(preds["predictions"], preds["brand"])
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

