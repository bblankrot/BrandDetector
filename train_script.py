import argparse
import spacy
import brand_detector
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train spaCy model for brand detection."
    )
    parser.add_argument(
        "dir",
        type=str,
        help='path of JSON datafile, containing "brand" and "transcription" pairs.',
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help=(
            "base model name. Leaving this empty usually leads to optimal behavior, "
            "but you can also try en_core_web_XX (XX=sm, md, or lg), provided you install them first."
        ),
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="output model directory",
        default=str("models"),
    )
    parser.add_argument(
        "-n", "--name", type=str, help="output model name", default="brand"
    )
    parser.add_argument(
        "-i", "--iters", type=int, help="output model directory", default=30
    )
    parser.add_argument(
        "-p",
        "--pct",
        type=float,
        help="fraction of training data from entire labeled dataset",
        default=0.8,
    )
    parser.add_argument(
        "-v",
        "--validate",
        help=("calculate accuracy on test set during training"),
        action="store_true",
    )
    parser.add_argument(
        "-a",
        "--apostrophe",
        help=("remove apostrophes from brand and transcription"),
        action="store_true",
    )
    parser.add_argument(
        "-s", "--synth", type=int, help="number of synthesized transcripts", default=0
    )
    parser.add_argument("--data", help=("add additional data"), default=None)
    args = parser.parse_args()

    has_gpu = spacy.prefer_gpu()
    print("Using GPU" if has_gpu else "No GPU, training on CPU")

    df_train, df_test, df_dirty = brand_detector.utils.generate_train_test_set(
        args.dir,
        args.pct,
        args.apostrophe,
        additional_data=args.data,
        n_synth=args.synth,
    )
    entity_list = brand_detector.train_spacy.df_to_entity_list(df_train)
    df_test.to_json(Path(args.output) / "test_data.json")
    #df_dirty.to_json("data/preprocessed/dirty_data.json")
    brand_detector.train_spacy.train_spacy(
        entity_list,
        model=args.model,
        new_model_name=args.name,
        output_dir=args.output,
        n_iter=args.iters,
        val=(df_test if args.validate else None),
    )
