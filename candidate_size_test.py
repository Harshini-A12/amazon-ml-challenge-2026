import pandas as pd
import re
import unicodedata


BASE = "Dataset"

S1_PATH = f"{BASE}/train/train_source1.tsv"
S2_PATH = f"{BASE}/train/train_source2.tsv"
S3_PATH = f"{BASE}/train/train_source3.tsv"


print("Loading data...")

s1 = pd.read_csv(S1_PATH, sep="\t", nrows=10000)
s2 = pd.read_csv(S2_PATH, sep="\t", nrows=10000)
s3 = pd.read_csv(S3_PATH, sep="\t", nrows=10000)

sources = pd.concat([s2, s3], ignore_index=True)


def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = "".join(
        " " if unicodedata.category(char).startswith("P")
        else char
        for char in text
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


STOPWORDS = {
    "inc", "incorporated",
    "llc",
    "ltd", "limited",
    "corp", "corporation",
    "company", "co",
    "private", "pvt",
    "plc"
}


def name_tokens(text):

    return {
        token
        for token in normalize_text(text).split()
        if len(token) > 1
        and token not in STOPWORDS
    }


def address_tokens(text):

    return {
        token
        for token in normalize_text(text).split()
        if len(token) > 1
    }


# ============================================================
# BUILD BLOCKING INDEXES
# ============================================================

print("\nBuilding indexes...")

name_index = {}
address_index = {}

for idx, row in sources.iterrows():

    country = row["country"]

    for token in name_tokens(row["business_name"]):

        key = (country, token)

        if key not in name_index:
            name_index[key] = set()

        name_index[key].add(idx)

    for token in address_tokens(row["business_address"]):

        key = (country, token)

        if key not in address_index:
            address_index[key] = set()

        address_index[key].add(idx)


# ============================================================
# CANDIDATE SIZE TEST
# ============================================================

print("\nTesting candidate sizes...")

sizes = []

for count, (_, row) in enumerate(
    s1.iterrows(),
    start=1
):

    country = row["country"]

    name_candidates = set()
    address_candidates = set()

    # Name route
    for token in name_tokens(row["business_name"]):

        key = (country, token)

        if key in name_index:
            name_candidates.update(
                name_index[key]
            )

    # Address route
    for token in address_tokens(row["business_address"]):

        key = (country, token)

        if key in address_index:
            address_candidates.update(
                address_index[key]
            )

    union_candidates = (
        name_candidates |
        address_candidates
    )

    sizes.append(len(union_candidates))

    if count % 1000 == 0:

        print(
            f"Processed {count}/10000"
        )


# ============================================================
# SUMMARY
# ============================================================

sizes = pd.Series(sizes)

print("\n========================================")
print("CANDIDATE SIZE SUMMARY")
print("========================================")

print("Average:", round(sizes.mean(), 2))
print("Median:", int(sizes.median()))
print("Minimum:", int(sizes.min()))
print("Maximum:", int(sizes.max()))

print("\nPercentiles:")

print(
    sizes.quantile(
        [0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

print("\nCandidate counts > thresholds:")

for threshold in [100, 500, 1000, 2000, 5000, 10000]:

    count = (sizes > threshold).sum()

    print(
        f"> {threshold}: {count} entities"
    )

print("\nDone.")