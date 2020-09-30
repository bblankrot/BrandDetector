import argparse
import spacy
import brand_detector
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train spaCy model for brand detection."
    )
    parser.add_argument("-m", "--model", type=str, help="base model name", default=None)
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
    parser.add_argument(
        "-a",
        "--apostrophe",
        help=("remove apostrophes from brand and transcription"),
        action="store_true",
    )
    args = parser.parse_args()

    if args.gpu:
        spacy.require_gpu()
    else:
        has_gpu = spacy.prefer_gpu()
        print("Using GPU" if has_gpu else "No GPU, training on CPU")

    df_train, df_test, df_dirty = brand_detector.utils.generate_train_test_set(
        "data/raw/listen_demo_records.json", args.pct, args.apostrophe
    )
    entity_list = brand_detector.train_spacy.df_to_entity_list(df_train)
    df_test.to_json("data/preprocessed/test_data.json")
    df_dirty.to_json("data/preprocessed/dirty_data.json")
    brand_detector.train_spacy.train_spacy(
        entity_list,
        model=args.model,
        new_model_name=args.name,
        output_dir=args.output,
        n_iter=args.iters,
        val=(df_test if args.validate else None),
    )
