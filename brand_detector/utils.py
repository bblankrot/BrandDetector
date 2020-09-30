import re
import pandas as pd


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
    df2 = df.copy()
    df2["brand_matches"] = df2.apply(
        lambda row: find_brands(row["brand"], row["transcription"]), axis=1
    )
    return df2


def temporary_credit_union_fix(row):
    if row["brand"].lower() != "credit union":
        return row["brand"]
    possible_unions = ["Belco", "Navy Federal", "Affinity", "Apple Federal", "Midland"]
    for cu in possible_unions:
        if (
            re.search(
                "\\b" + cu + " Credit Union\\b", row["transcription"], re.IGNORECASE
            )
            is not None
        ):
            # return only first, since this is a temporary fix, edge cases do not matter as much
            return cu + " Credit Union"
    return "Credit Union"


def preprocess(df, deapostrophe=False):
    # remove rows with no brand
    df = df[(df["brand"].notna()) & (df["brand"] != "")].copy()
    # remove dealership ranking
    df.loc[:, "brand"] = df["brand"].str.replace(r"-tier[1-3]$", "", regex=True)
    # remove incidental mistakes
    df.loc[df["brand"] == "Zico", "brand"] = "Geico"
    df.loc[df["brand"].str.lower() == "the home depot", "brand"] = "Home Depot"
    df.loc[df["brand"] == "t", "brand"] = "T-Mobile"
    # remove leading/trailing whitespace
    df.loc[:, "brand"] = df["brand"].str.strip()
    # temporarily hack together some credit union labels, and remove non-obvious ones
    df.loc[:, "brand"] = df.apply(temporary_credit_union_fix, axis=1)
    df = df[df["brand"].str.lower() != "credit union"].copy()
    if deapostrophe:
        for col in ["brand", "transcription"]:
            df.loc[:, col] = df[col].str.replace("'", "")
    return df


def generate_train_test_set(filename, pct, deapostrophe):
    df = pd.read_json(filename)
    df = preprocess(df, deapostrophe=deapostrophe)
    df = find_brands_in_df(df)
    df_train_test = df[df["brand_matches"].str.len() > 0]
    num_rows = df_train_test.shape[0]
    num_train = int(round(num_rows * pct))
    df_train, df_test = (
        df_train_test.iloc[:num_train, :].copy(),
        df_train_test.iloc[num_train:, :].copy(),
    )
    df_dirty = df[df["brand_matches"].str.len() == 0].copy()

    seen_set = set(
        df_train["brand"].str.lower().to_numpy()
    )  # has this brand been seen as is?
    df_test["seen_in_training"] = df_test["brand"].apply(
        lambda x: x.lower() in seen_set
    )
    # not strictly necessary, since dirty dataset does not have brand name in transcript, but might be interesting
    df_dirty["seen_in_training"] = df_dirty["brand"].apply(
        lambda x: x.lower() in seen_set
    )
    return df_train, df_test, df_dirty
