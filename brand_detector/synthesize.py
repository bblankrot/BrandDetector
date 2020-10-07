from gibberish import Gibberish
from faker import Faker
import random
import re

def generate_synthetic_names(n_synth):
    n_faker = min(round(n_synth / 2), 700)  # reduce repetitions, faker come from limited dataset
    n_gibberish = n_synth - n_faker
    synth_list = []

    fake = Faker()
    for _ in range(n_faker):
        # faker returns brands composed of multiple last names, just return first (may include hyphens)
        name = fake.company().replace(",", "").split()[0]
        if random.random() < 0.5:
            name = name.lower()
        if ("-" in name) and (random.random() < 0.5):
            name = name.replace("-", " ")
        synth_list.append(name)

    gib = Gibberish()
    for _ in range(n_gibberish):
        # gibberish returns uncapitalized gibberish, by default begins and ends with consonants
        type_of_gibberish = random.random()
        if type_of_gibberish < 0.25:
            name = gib.generate_word(start_vowel=True)
        elif type_of_gibberish < 0.5:
            name = gib.generate_word(end_vowel=True)
        elif type_of_gibberish < 0.75:
            name = gib.generate_word()
        else:
            name = gib.generate_word(2, start_vowel=True, end_vowel=True)
        add_gibberish = random.random()
        if add_gibberish < 0.1:
            name += " " + gib.generate_word()
        if random.random() < 0.5:
            name = name.title()
        synth_list.append(name)

    random.shuffle(synth_list)
    return synth_list


def generate_synthetic_df(df, n_synth=None):
    n_rows = df.shape[0]
    if n_synth is None:
        n_synth = round(n_rows / 10)
    else:
        n_synth = min(n_synth, n_rows)
    
    synth_list = generate_synthetic_names(n_synth)
    df_synth = df[["brand", "transcription"]].sample(n_synth).copy()

    for (_, row), synth_name in zip(df_synth.iterrows(), synth_list):
        row["transcription"] = re.sub(
            r"\b" + re.escape(row["brand"]) + r"\b",
            synth_name,
            row["transcription"],
            flags=re.IGNORECASE,
        )
        row["brand"] = synth_name

    df_synth.index = [f"s{ind}" for ind in df_synth.index]
    return df_synth
