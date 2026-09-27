import pandas as pd
import numpy as np
import re
import unicodedata

from rapidfuzz import fuzz


# ============================================================
# FILES
# ============================================================

POSITIVE_FILE = "training_positive_pairs.csv"
NEGATIVE_FILE = "training_hard_negatives.csv"

S1_FILE = "Dataset/train/train_source1.tsv"
S2_FILE = "Dataset/train/train_source2.tsv"
S3_FILE = "Dataset/train/train_source3.tsv"

OUTPUT_FILE = "training_features.csv"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = "".join(
        " " if unicodedata.category(c).startswith("P") else c
        for c in text
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_tokens(text):

    text = normalize_text(text)

    if not text:
        return set()

    return {
        token
        for token in text.split()
        if len(token) > 1
    }


# ============================================================
# TOKEN OVERLAP
# ============================================================

def token_overlap(tokens_a, tokens_b):

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / min(
        len(tokens_a),
        len(tokens_b)
    )


# ============================================================
# LOAD TRAINING PAIRS
# ============================================================

print("Loading training pairs...")

positive = pd.read_csv(POSITIVE_FILE)
negative = pd.read_csv(NEGATIVE_FILE)

pairs = pd.concat(
    [positive, negative],
    ignore_index=True
)

print("Positive pairs:", len(positive))
print("Negative pairs:", len(negative))
print("Total pairs:", len(pairs))


# ============================================================
# WHICH SOURCE DOES MATCHED ID BELONG TO?
# ============================================================

pairs["source"] = pairs["matched_entity_id"].astype(str).str[:2]

print("\nMatched-source distribution:")
print(pairs["source"].value_counts())


# ============================================================
# LOAD S1
# ============================================================

print("\nLoading Source 1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

s1 = s1.rename(
    columns={
        "entity_id": "source1_entity_id",
        "business_name": "s1_name",
        "business_address": "s1_address",
        "country": "s1_country"
    }
)

s1 = s1[
    s1["source1_entity_id"].isin(
        pairs["source1_entity_id"]
    )
]

print("S1 rows:", len(s1))


# ============================================================
# LOAD SOURCE 2
# ============================================================

print("\nLoading Source 2...")

needed_s2_ids = set(
    pairs.loc[
        pairs["source"] == "S2",
        "matched_entity_id"
    ]
)

s2_parts = []

for chunk in pd.read_csv(
    S2_FILE,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ],
    chunksize=100000
):

    chunk = chunk[
        chunk["entity_id"].isin(needed_s2_ids)
    ]

    if len(chunk) > 0:
        s2_parts.append(chunk)

s2 = pd.concat(
    s2_parts,
    ignore_index=True
)

s2 = s2.rename(
    columns={
        "entity_id": "matched_entity_id",
        "business_name": "s2_name",
        "business_address": "s2_address",
        "country": "s2_country"
    }
)

print("S2 rows loaded:", len(s2))


# ============================================================
# LOAD SOURCE 3
# ============================================================

print("\nLoading Source 3...")

needed_s3_ids = set(
    pairs.loc[
        pairs["source"] == "S3",
        "matched_entity_id"
    ]
)

s3_parts = []

for chunk in pd.read_csv(
    S3_FILE,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ],
    chunksize=100000
):

    chunk = chunk[
        chunk["entity_id"].isin(needed_s3_ids)
    ]

    if len(chunk) > 0:
        s3_parts.append(chunk)

s3 = pd.concat(
    s3_parts,
    ignore_index=True
)

s3 = s3.rename(
    columns={
        "entity_id": "matched_entity_id",
        "business_name": "s3_name",
        "business_address": "s3_address",
        "country": "s3_country"
    }
)

print("S3 rows loaded:", len(s3))


# ============================================================
# MERGE S1 INFORMATION
# ============================================================

print("\nMerging S1...")

pairs = pairs.merge(
    s1,
    on="source1_entity_id",
    how="left"
)


# ============================================================
# MERGE S2
# ============================================================

pairs = pairs.merge(
    s2,
    on="matched_entity_id",
    how="left"
)


# ============================================================
# MERGE S3
# ============================================================

pairs = pairs.merge(
    s3,
    on="matched_entity_id",
    how="left"
)


# ============================================================
# SELECT MATCHED RECORD
# ============================================================

pairs["match_name"] = pairs["s2_name"].fillna(
    pairs["s3_name"]
)

pairs["match_address"] = pairs["s2_address"].fillna(
    pairs["s3_address"]
)

pairs["match_country"] = pairs["s2_country"].fillna(
    pairs["s3_country"]
)


# ============================================================
# FEATURE CALCULATION
# ============================================================

print("\nCalculating features...")

features = []

for i, row in enumerate(
    pairs.itertuples(index=False),
    start=1
):

    s1_name = normalize_text(row.s1_name)
    match_name = normalize_text(row.match_name)

    s1_address = normalize_text(row.s1_address)
    match_address = normalize_text(row.match_address)

    name_tokens_1 = get_tokens(s1_name)
    name_tokens_2 = get_tokens(match_name)

    address_tokens_1 = get_tokens(s1_address)
    address_tokens_2 = get_tokens(match_address)

    # -------------------------------
    # NAME FEATURES
    # -------------------------------

    name_wratio = fuzz.WRatio(
        s1_name,
        match_name
    )

    name_token_sort = fuzz.token_sort_ratio(
        s1_name,
        match_name
    )

    name_token_set = fuzz.token_set_ratio(
        s1_name,
        match_name
    )

    name_overlap = token_overlap(
        name_tokens_1,
        name_tokens_2
    )

    # -------------------------------
    # ADDRESS FEATURES
    # -------------------------------

    address_missing = (
        int(
            not s1_address
            or not match_address
        )
    )

    if address_missing:

        address_wratio = 0.0
        address_token_sort = 0.0
        address_token_set = 0.0
        address_overlap = 0.0

    else:

        address_wratio = fuzz.WRatio(
            s1_address,
            match_address
        )

        address_token_sort = fuzz.token_sort_ratio(
            s1_address,
            match_address
        )

        address_token_set = fuzz.token_set_ratio(
            s1_address,
            match_address
        )

        address_overlap = token_overlap(
            address_tokens_1,
            address_tokens_2
        )

    # -------------------------------
    # COUNTRY
    # -------------------------------

    country_equal = int(
        row.s1_country == row.match_country
    )

    features.append({

        "source1_entity_id":
            row.source1_entity_id,

        "matched_entity_id":
            row.matched_entity_id,

        "name_wratio":
            name_wratio,

        "name_token_sort":
            name_token_sort,

        "name_token_set":
            name_token_set,

        "name_token_overlap":
            name_overlap,

        "address_wratio":
            address_wratio,

        "address_token_sort":
            address_token_sort,

        "address_token_set":
            address_token_set,

        "address_token_overlap":
            address_overlap,

        "address_missing":
            address_missing,

        "country_equal":
            country_equal,

        "label":
            row.label
    })

    if i % 10000 == 0:
        print(
            "Processed pairs:",
            i,
            "/",
            len(pairs)
        )


# ============================================================
# SAVE
# ============================================================

feature_df = pd.DataFrame(features)

print("\n============================================")
print("FEATURE GENERATION COMPLETE")
print("============================================")

print("Rows:", len(feature_df))

print("\nClass distribution:")
print(feature_df["label"].value_counts())

feature_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:", OUTPUT_FILE)