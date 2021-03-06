import re
import pandas as pd
from .synthesize import generate_synthetic_df


def find_brands(brand, text, ignore_case=True, dehyphenate=True):
    """Find occurences of a given correct brand in a transcription, up to some deviations."""
    if brand == "":
        return []
    brand_d = brand.replace("-", " ") if dehyphenate else brand
    text_d = text.replace("-", " ") if dehyphenate else text
    # if there's punctuation at the beginning or end of the brand, remove it
    inner_brand = re.match(r"([a-zA-Z0-9()].*[a-zA-Z0-9()])", brand_d)
    if inner_brand is None:
        raise RuntimeError("encountered empty brand: {}".format(brand))
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
    """Finds brands in each transcription and appends them as a new column."""
    df2 = df.copy()
    df2["brand_matches"] = df2.apply(
        lambda row: find_brands(row["brand"], row["transcription"]), axis=1
    )
    return df2


def preprocess(df, deapostrophe=False):
    """Apply preprocessing rules to clean training/val data."""
    # remove rows with no brand
    df = df[(df["brand"].notna()) & (df["brand"] != "")].copy()
    # remove leading/trailing whitespace
    df.loc[:, "brand"] = df["brand"].str.strip()
    if deapostrophe:
        for col in ["brand", "transcription"]:
            df.loc[:, col] = df[col].str.replace("'", "")
    return df


def generate_additional_data(additional_data, deapostrophe):
    """Add additional external dataset to training data, if it exists."""
    df_add = pd.read_json(additional_data)
    if deapostrophe:
        for col in ["brand", "transcription"]:
            df_add.loc[:, col] = df_add[col].str.replace("'", "")
    df_add = find_brands_in_df(df_add)
    df_add = df_add[df_add["brand_matches"].apply(len) > 0]

    df_add.index = [f"n{ind}" for ind in df_add.index]
    return df_add


def generate_lower(df_train):
    """Create copy of dataset, consisting of lowercased brands and transcriptions."""
    df_train_lower = df_train[["brand", "transcription", "brand_matches"]].copy()
    for col in ["brand", "transcription"]:
        df_train_lower[col] = df_train_lower[col].str.lower()
    df_train_lower.index = [f"l{ind}" for ind in df_train_lower.index]
    return df_train_lower


def generate_train_test_set(
    filename, pct, deapostrophe, additional_data=None, n_synth=0, lowercase=True
):
    """Load a dataset, apply preprocessing, and split into training and validation sets."""
    df = pd.read_json(filename)
    df = preprocess(df, deapostrophe=deapostrophe)
    df = find_brands_in_df(df)
    df_train_test = df[df["brand_matches"].apply(len) > 0].sample(
        frac=1, random_state=1
    )
    num_rows = df_train_test.shape[0]
    num_train = int(round(num_rows * pct))
    df_train, df_test = (
        df_train_test.iloc[:num_train, :].copy(),
        df_train_test.iloc[num_train:, :].copy(),
    )
    df_dirty = df[df["brand_matches"].apply(len) == 0].copy()

    if lowercase:
        df_lower = generate_lower(df_train)
        df_train = pd.concat([df_train, df_lower], axis=0)

    if additional_data is not None:
        df_add = generate_additional_data(additional_data, deapostrophe)
        df_train = pd.concat([df_train, df_add], axis=0)

    if n_synth > 0:
        df_synth = generate_synthetic_df(df_train, n_synth=n_synth)
        df_synth = find_brands_in_df(df_synth)
        df_train = pd.concat([df_train, df_synth], axis=0)

    seen_set = set(
        df_train["brand"].str.lower().to_numpy()
    )  # has this brand been seen as is?
    df_test["seen_in_training"] = df_test["brand"].apply(
        lambda x: x.lower() in seen_set
    )
    # not strictly necessary, since dirty dataset does not have brand name in
    # transcript
    df_dirty["seen_in_training"] = df_dirty["brand"].apply(
        lambda x: x.lower() in seen_set
    )
    return df_train, df_test, df_dirty
