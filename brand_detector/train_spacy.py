import pandas as pd
import numpy as np
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import json
import time
import statistics

from .utils import find_brands
from .predict import predict_df, preds2corrects


def df_to_entity_list(df):
    """Convert found matches to the entity list form that spacy accepts."""
    train = []
    for _, row in df.iterrows():
        if not row["brand_matches"]:
            continue  # should not happen
        matches = []
        for match in row["brand_matches"]:
            matches.append((match[0], match[1], "BRAND"))
        train.append((row["transcription"], {"entities": matches}))
    return train


def train_spacy(
    entity_list,
    model=None,
    new_label="BRAND",
    new_model_name="brand",
    output_dir=str(Path.home() / "models"),
    n_iter=30,
    val=None,
):
    """Set up the pipeline and entity recognizer, and train the new entity, BRAND.
    If a validation set is provided, compute the accuracy and save the model after
    each epoch, returning the best model # at the end."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        #print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        #print("Created blank 'en' model")
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
            val_acc = 0
            random.shuffle(entity_list)
            batches = minibatch(entity_list, size=sizes)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            cur_time = time.time()
            print("Losses", losses)
            print("({:.2f} seconds)".format(cur_time - start_time))
            start_time = cur_time

            if val is not None:
                # save to disk after each iteration
                epoch_path = output_dir / "{0}_epoch_{1}".format(new_model_name, i)
                nlp.to_disk(epoch_path)
                preds = predict_df(nlp, val)
                _, val_acc = preds2corrects(preds["predictions"], preds["brand"])
                print("Val accuracy", val_acc)
            history.append({"losses": losses, "val_accuracy": val_acc})

    # save model to output directory
    nlp.meta["name"] = new_model_name  # rename model
    nlp.to_disk(output_dir / new_model_name)
    print("Saved model to", output_dir)

    if val is not None:
        vals = np.array([hist["val_accuracy"] for hist in history])
        argmax_val = vals.argmax()
        print(
            "Best accuracy ({:.2f}) achieved at epoch {}".format(
                vals[argmax_val], argmax_val
            )
        )

    with open(output_dir / new_model_name / "history.json", "w") as fp:
        json.dump(history, fp)

    return nlp

