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
# 2. LOAD PROTOTYPE DATA
# ============================================================

print("Loading data...")

s1 = pd.read_csv(S1_PATH, sep="\t", nrows=10000)
s2 = pd.read_csv(S2_PATH, sep="\t", nrows=10000)
s3 = pd.read_csv(S3_PATH, sep="\t", nrows=10000)

gt = pd.read_csv(GT_PATH, sep="\t")

sources = pd.concat([s2, s3], ignore_index=True)

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


def useful_name_tokens(text):

    text = normalize_text(text)

    return {
        token
        for token in text.split()
        if len(token) > 1
        and token not in STOPWORDS
    }


def address_tokens(text):

    text = normalize_text(text)

    return {
        token
        for token in text.split()
        if len(token) > 1
    }


# ============================================================
# 4. SIMILARITY FUNCTIONS
# ============================================================

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
# 5. PREPARE LOOKUPS
# ============================================================

s1_lookup = s1.set_index("entity_id")
source_lookup = sources.set_index("entity_id")


# ============================================================
# 6. CREATE BLOCKING INDEXES
# ============================================================

print("\nBuilding blocking indexes...")

name_index = {}
address_index = {}

for idx, row in sources.iterrows():

    entity_id = row["entity_id"]
    country = row["country"]

    # Name tokens
    name_tokens = useful_name_tokens(
        row["business_name"]
    )

    for token in name_tokens:

        key = (country, token)

        if key not in name_index:
            name_index[key] = set()

        name_index[key].add(idx)


    # Address tokens
    addr_tokens = address_tokens(
        row["business_address"]
    )

    for token in addr_tokens:

        key = (country, token)

        if key not in address_index:
            address_index[key] = set()

        address_index[key].add(idx)


print("Name index keys:", len(name_index))
print("Address index keys:", len(address_index))


# ============================================================
# 7. GROUND TRUTH LOOKUP
# ============================================================

ground_truth = {}

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    matched = str(
        row["matched_entity_ids"]
    )

    if matched == "nan" or matched == "":
        ground_truth[s1_id] = set()

    else:
        ground_truth[s1_id] = {
            x.strip()
            for x in matched.split(",")
            if x.strip()
        }


# ============================================================
# 8. GENERATE CANDIDATE PAIRS
# ============================================================

print("\nGenerating candidate pairs...")

feature_rows = []

MAX_S1 = 10000

for count, (_, row) in enumerate(
    s1.head(MAX_S1).iterrows(),
    start=1
):

    s1_id = row["entity_id"]
    country = row["country"]

    name_tokens = useful_name_tokens(
        row["business_name"]
    )

    addr_tokens = address_tokens(
        row["business_address"]
    )

    candidate_indexes = set()


    # ----------------------------
    # Name route
    # ----------------------------

    for token in name_tokens:

        key = (country, token)

        if key in name_index:
            candidate_indexes.update(
                name_index[key]
            )


    # ----------------------------
    # Address route
    # ----------------------------

    for token in addr_tokens:

        key = (country, token)

        if key in address_index:
            candidate_indexes.update(
                address_index[key]
            )


    true_matches = ground_truth.get(
        s1_id,
        set()
    )


    # ========================================================
    # 9. CREATE FEATURES
    # ========================================================

    for candidate_idx in candidate_indexes:

        candidate = sources.iloc[candidate_idx]

        candidate_id = candidate["entity_id"]


        # ----------------------------
        # Name similarities
        # ----------------------------

        name_wratio, name_sort, name_set = \
            text_similarities(
                row["business_name"],
                candidate["business_name"]
            )


        name_overlap = token_overlap(
            useful_name_tokens(
                row["business_name"]
            ),
            useful_name_tokens(
                candidate["business_name"]
            )
        )


        # ----------------------------
        # Address similarities
        # ----------------------------

        address_missing = int(
            pd.isna(row["business_address"])
            or
            pd.isna(candidate["business_address"])
        )


        if address_missing:

            address_wratio = 0.0
            address_sort = 0.0
            address_set = 0.0
            address_overlap = 0.0

        else:

            address_wratio, address_sort, address_set = \
                text_similarities(
                    row["business_address"],
                    candidate["business_address"]
                )

            address_overlap = token_overlap(
                address_tokens(
                    row["business_address"]
                ),
                address_tokens(
                    candidate["business_address"]
                )
            )


        # ----------------------------
        # Country
        # ----------------------------

        country_equal = int(
            row["country"] == candidate["country"]
        )


        # ----------------------------
        # Label
        # ----------------------------

        label = int(
            candidate_id in true_matches
        )


        # ----------------------------
        # Save row
        # ----------------------------

        feature_rows.append({

            "source1_entity_id": s1_id,
            "candidate_entity_id": candidate_id,

            "name_wratio": name_wratio,
            "name_token_sort": name_sort,
            "name_token_set": name_set,
            "name_token_overlap": name_overlap,

            "address_wratio": address_wratio,
            "address_token_sort": address_sort,
            "address_token_set": address_set,
            "address_token_overlap": address_overlap,

            "address_missing": address_missing,

            "country_equal": country_equal,

            "label": label
        })


    if count % 1000 == 0:

        print(
            f"Processed {count}/{MAX_S1} S1 entities | "
            f"candidate pairs: {len(feature_rows):,}"
        )


# ============================================================
# 10. CREATE FEATURE DATAFRAME
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


# ============================================================
# 11. SAVE
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