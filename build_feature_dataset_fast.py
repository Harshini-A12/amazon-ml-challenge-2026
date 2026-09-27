import pandas as pd
import re
import unicodedata
from rapidfuzz import fuzz


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE = "Dataset"

S1_PATH = f"{BASE}/train/train_source1.tsv"
S2_PATH = f"{BASE}/train/train_source2.tsv"
S3_PATH = f"{BASE}/train/train_source3.tsv"
GT_PATH = f"{BASE}/train/train_ground_truth.tsv"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading data...")

s1 = pd.read_csv(S1_PATH, sep="\t", nrows=10000)
s2 = pd.read_csv(S2_PATH, sep="\t", nrows=10000)
s3 = pd.read_csv(S3_PATH, sep="\t", nrows=10000)
gt = pd.read_csv(GT_PATH, sep="\t")

sources = pd.concat(
    [s2, s3],
    ignore_index=True
)

print("S1 rows:", len(s1))
print("S2 + S3 rows:", len(sources))


# ============================================================
# 3. NORMALIZATION
# ============================================================

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


def token_overlap(tokens1, tokens2):

    if not tokens1 or not tokens2:
        return 0.0

    return len(tokens1 & tokens2) / min(
        len(tokens1),
        len(tokens2)
    )


def text_similarities(text1, text2):

    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    if not text1 or not text2:
        return 0.0, 0.0, 0.0

    return (
        fuzz.WRatio(text1, text2) / 100,
        fuzz.token_sort_ratio(text1, text2) / 100,
        fuzz.token_set_ratio(text1, text2) / 100
    )


# ============================================================
# 4. PRECOMPUTE SOURCE INFORMATION
# ============================================================

print("\nPreparing source records...")

source_records = {}

for _, row in sources.iterrows():

    entity_id = row["entity_id"]

    source_records[entity_id] = {
        "business_name": row["business_name"],
        "business_address": row["business_address"],
        "country": row["country"],
        "name_tokens": name_tokens(
            row["business_name"]
        ),
        "address_tokens": address_tokens(
            row["business_address"]
        )
    }


# ============================================================
# 5. BUILD BLOCKING INDEXES
# ============================================================

print("Building blocking indexes...")

name_index = {}
address_index = {}

for entity_id, record in source_records.items():

    country = record["country"]

    for token in record["name_tokens"]:

        key = (country, token)

        name_index.setdefault(
            key,
            set()
        ).add(entity_id)

    for token in record["address_tokens"]:

        key = (country, token)

        address_index.setdefault(
            key,
            set()
        ).add(entity_id)


print("Name index keys:", len(name_index))
print("Address index keys:", len(address_index))


# ============================================================
# 6. GROUND TRUTH LOOKUP
# ============================================================

print("Preparing ground truth...")

ground_truth = {}

for _, row in gt.iterrows():

    matched = str(
        row["matched_entity_ids"]
    )

    if matched == "nan" or matched == "":

        ground_truth[
            row["source1_entity_id"]
        ] = set()

    else:

        ground_truth[
            row["source1_entity_id"]
        ] = {
            x.strip()
            for x in matched.split(",")
            if x.strip()
        }


# ============================================================
# 7. BUILD FEATURES
# ============================================================

print("\nGenerating feature dataset...")

feature_rows = []

for count, (_, row) in enumerate(
    s1.iterrows(),
    start=1
):

    s1_id = row["entity_id"]
    country = row["country"]

    s1_name_tokens = name_tokens(
        row["business_name"]
    )

    s1_address_tokens = address_tokens(
        row["business_address"]
    )


    # --------------------------------------------------------
    # ORIGINAL BLOCKING
    # Country + (name OR address)
    # --------------------------------------------------------

    candidates = set()

    for token in s1_name_tokens:

        candidates.update(
            name_index.get(
                (country, token),
                set()
            )
        )

    for token in s1_address_tokens:

        candidates.update(
            address_index.get(
                (country, token),
                set()
            )
        )


    true_matches = ground_truth.get(
        s1_id,
        set()
    )


    # --------------------------------------------------------
    # CHEAP FILTER
    # Keep candidate if:
    #
    # shared name >= 1
    # OR
    # shared address >= 1
    # --------------------------------------------------------

    for candidate_id in candidates:

        candidate = source_records[
            candidate_id
        ]

        shared_name = len(
            s1_name_tokens &
            candidate["name_tokens"]
        )

        shared_address = len(
            s1_address_tokens &
            candidate["address_tokens"]
        )


        if (
            shared_name < 1
            and
            shared_address < 1
        ):
            continue


        # ----------------------------------------------------
        # NAME FEATURES
        # ----------------------------------------------------

        (
            name_wratio,
            name_token_sort,
            name_token_set
        ) = text_similarities(
            row["business_name"],
            candidate["business_name"]
        )


        name_overlap = token_overlap(
            s1_name_tokens,
            candidate["name_tokens"]
        )


        # ----------------------------------------------------
        # ADDRESS FEATURES
        # ----------------------------------------------------

        address_missing = int(
            pd.isna(row["business_address"])
            or
            pd.isna(candidate["business_address"])
        )


        if address_missing:

            address_wratio = 0.0
            address_token_sort = 0.0
            address_token_set = 0.0
            address_overlap = 0.0

        else:

            (
                address_wratio,
                address_token_sort,
                address_token_set
            ) = text_similarities(
                row["business_address"],
                candidate["business_address"]
            )

            address_overlap = token_overlap(
                s1_address_tokens,
                candidate["address_tokens"]
            )


        # ----------------------------------------------------
        # COUNTRY
        # ----------------------------------------------------

        country_equal = int(
            country == candidate["country"]
        )


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = int(
            candidate_id in true_matches
        )


        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        feature_rows.append({

            "source1_entity_id": s1_id,
            "candidate_entity_id": candidate_id,

            "name_wratio": name_wratio,
            "name_token_sort": name_token_sort,
            "name_token_set": name_token_set,
            "name_token_overlap": name_overlap,

            "address_wratio": address_wratio,
            "address_token_sort": address_token_sort,
            "address_token_set": address_token_set,
            "address_token_overlap": address_overlap,

            "address_missing": address_missing,
            "country_equal": country_equal,

            "label": label
        })


    if count % 1000 == 0:

        print(
            f"Processed {count}/10000 | "
            f"feature rows: {len(feature_rows):,}"
        )


# ============================================================
# 8. CREATE DATAFRAME
# ============================================================

features = pd.DataFrame(feature_rows)

print("\n========================================")
print("FEATURE DATASET CREATED")
print("========================================")

print("Rows:", len(features))
print("Columns:", len(features.columns))


print("\nClass distribution:")

print(
    features["label"]
    .value_counts()
)


print("\nPositive percentage:")

print(
    round(
        features["label"].mean() * 100,
        4
    ),
    "%"
)


# ============================================================
# 9. SAVE
# ============================================================

OUTPUT_PATH = "prototype_features.csv"

features.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:", OUTPUT_PATH)

print("\nFirst 5 rows:")

print(
    features.head()
)

print("\nDone.")