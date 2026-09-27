import pandas as pd
import numpy as np
import re
import unicodedata
import joblib

from rapidfuzz import fuzz


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_S1 = 10000
RANDOM_STATE = 42

S1_FILE = "Dataset/train/train_source1.tsv"
S2_FILE = "Dataset/train/train_source2.tsv"
S3_FILE = "Dataset/train/train_source3.tsv"
GT_FILE = "Dataset/train/train_ground_truth.tsv"

MODEL_FILE = "matching_model.pkl"
THRESHOLD_FILE = "matching_threshold.txt"


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


def token_overlap(a, b):

    if not a or not b:
        return 0.0

    return len(a & b) / min(len(a), len(b))


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("Loading model...")

model = joblib.load(MODEL_FILE)

with open(THRESHOLD_FILE) as f:
    threshold = float(f.read().strip())

print("Threshold:", threshold)


# ============================================================
# 2. LOAD S1
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

s1 = s1.sample(
    n=min(SAMPLE_S1, len(s1)),
    random_state=RANDOM_STATE
).reset_index(drop=True)

print("S1 sample:", len(s1))


# ============================================================
# 3. LOAD SOURCE 2 + SOURCE 3
# ============================================================

print("\nLoading Source 2...")

s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print("S2 rows:", len(s2))


print("\nLoading Source 3...")

s3 = pd.read_csv(
    S3_FILE,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print("S3 rows:", len(s3))


# ============================================================
# 4. COMBINE SOURCES
# ============================================================

sources = pd.concat(
    [s2, s3],
    ignore_index=True
)

print("\nCombined candidate records:", len(sources))


# ============================================================
# 5. PRECOMPUTE TOKENS
# ============================================================

print("\nPreparing tokens...")

s1["name_tokens"] = s1["business_name"].map(get_tokens)
s1["address_tokens"] = s1["business_address"].map(get_tokens)

sources["name_tokens"] = sources["business_name"].map(get_tokens)
sources["address_tokens"] = sources["business_address"].map(get_tokens)

print("Token preparation complete.")


# ============================================================
# 6. BUILD TOKEN INDEX
# ============================================================

print("\nBuilding token index...")

name_index = {}
address_index = {}

for i, row in sources.iterrows():

    for token in row["name_tokens"]:

        if token not in name_index:
            name_index[token] = []

        name_index[token].append(i)

    for token in row["address_tokens"]:

        if token not in address_index:
            address_index[token] = []

        address_index[token].append(i)

print("Name tokens:", len(name_index))
print("Address tokens:", len(address_index))


# ============================================================
# 7. LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t"
)

gt = gt[
    gt["source1_entity_id"].isin(
        set(s1["entity_id"])
    )
].copy()

ground_truth = {}

for row in gt.itertuples(index=False):

    value = row.matched_entity_ids

    if pd.isna(value) or str(value).strip() == "":
        ground_truth[row.source1_entity_id] = set()

    else:
        ground_truth[row.source1_entity_id] = {
            x.strip()
            for x in str(value).split(",")
        }

print("Ground-truth S1 entities:", len(ground_truth))


# ============================================================
# 8. CANDIDATE GENERATION + MODEL
# ============================================================

FEATURES = [
    "name_wratio",
    "name_token_sort",
    "name_token_set",
    "name_token_overlap",
    "address_wratio",
    "address_token_sort",
    "address_token_set",
    "address_token_overlap",
    "address_missing",
    "country_equal"
]

predictions = {}

total_candidates = 0
total_after_filter = 0

print("\nRunning end-to-end matching...")

for count, row in enumerate(
    s1.itertuples(index=False),
    start=1
):

    s1_id = row.entity_id
    country = row.country

    s1_name = normalize_text(row.business_name)
    s1_address = normalize_text(row.business_address)

    s1_name_tokens = row.name_tokens
    s1_address_tokens = row.address_tokens

    # --------------------------------------------------------
    # BLOCKING
    # --------------------------------------------------------

    candidate_indices = set()

    # Name route
    for token in s1_name_tokens:

        if token in name_index:

            candidate_indices.update(
                name_index[token]
            )

    # Address route
    for token in s1_address_tokens:

        if token in address_index:

            candidate_indices.update(
                address_index[token]
            )

    total_candidates += len(candidate_indices)

    # --------------------------------------------------------
    # SECOND-STAGE FILTER
    # --------------------------------------------------------

    filtered = []

    for idx in candidate_indices:

        candidate = sources.iloc[idx]

        # Country blocking
        if candidate["country"] != country:
            continue

        name_overlap_count = len(
            s1_name_tokens &
            candidate["name_tokens"]
        )

        address_overlap_count = len(
            s1_address_tokens &
            candidate["address_tokens"]
        )

        # Keep if:
        # name >= 2 OR address >= 2

        if (
            name_overlap_count >= 2
            or
            address_overlap_count >= 2
        ):

            filtered.append(idx)

    total_after_filter += len(filtered)

    # --------------------------------------------------------
    # FEATURE CALCULATION
    # --------------------------------------------------------

    if not filtered:

        predictions[s1_id] = set()

        continue

    feature_rows = []
    candidate_ids = []

    for idx in filtered:

        candidate = sources.iloc[idx]

        match_name = normalize_text(
            candidate["business_name"]
        )

        match_address = normalize_text(
            candidate["business_address"]
        )

        name_tokens = candidate["name_tokens"]
        address_tokens = candidate["address_tokens"]

        # Name features
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

        name_token_overlap = token_overlap(
            s1_name_tokens,
            name_tokens
        )

        # Address features
        address_missing = int(
            not s1_address
            or not match_address
        )

        if address_missing:

            address_wratio = 0.0
            address_token_sort = 0.0
            address_token_set = 0.0
            address_token_overlap = 0.0

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

            address_token_overlap = token_overlap(
                s1_address_tokens,
                address_tokens
            )

        country_equal = int(
            country == candidate["country"]
        )

        feature_rows.append([
            name_wratio,
            name_token_sort,
            name_token_set,
            name_token_overlap,
            address_wratio,
            address_token_sort,
            address_token_set,
            address_token_overlap,
            address_missing,
            country_equal
        ])

        candidate_ids.append(
            candidate["entity_id"]
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    feature_matrix = pd.DataFrame(
        feature_rows,
        columns=FEATURES
    )

    probabilities = model.predict_proba(
        feature_matrix
    )[:, 1]

    matched_ids = {
        candidate_ids[i]
        for i, probability in enumerate(probabilities)
        if probability >= threshold
    }

    predictions[s1_id] = matched_ids

    if count % 100 == 0:

        print(
            "Processed:",
            count,
            "/",
            len(s1)
        )


# ============================================================
# 9. ENTITY-LEVEL F0.5
# ============================================================

print("\n")
print("=" * 70)
print("ENTITY-LEVEL EVALUATION")
print("=" * 70)


def entity_f05(predicted, actual):

    if not predicted and not actual:
        return 1.0

    if not predicted:
        return 0.0

    if not actual:
        return 0.0

    predicted = set(predicted)
    actual = set(actual)

    true_positive = len(
        predicted & actual
    )

    false_positive = len(
        predicted - actual
    )

    false_negative = len(
        actual - predicted
    )

    precision = (
        true_positive /
        (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive /
        (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    beta = 0.5

    if precision == 0 and recall == 0:
        return 0.0

    return (
        (1 + beta ** 2)
        * precision
        * recall
        /
        (
            beta ** 2 * precision
            + recall
        )
    )


scores = []

blocking_recall_values = []

for s1_id, actual in ground_truth.items():

    predicted = predictions.get(
        s1_id,
        set()
    )

    scores.append(
        entity_f05(
            predicted,
            actual
        )
    )

    # Blocking recall:
    # how many true matches survived candidate generation
    if actual:

        # We need to reconstruct candidates from
        # predictions + model output, so this is
        # evaluated through predicted matches here.
        recovered = len(
            predicted & actual
        )

        blocking_recall_values.append(
            recovered / len(actual)
        )


macro_f05 = np.mean(scores)

print("Entities evaluated:", len(scores))

print(
    "Macro entity-level F0.5:",
    round(macro_f05, 6)
)

print(
    "Average recovered-match ratio:",
    round(
        np.mean(blocking_recall_values),
        6
    )
)

print("\nCandidate statistics:")

print(
    "Average initial candidates:",
    round(
        total_candidates / len(s1),
        2
    )
)

print(
    "Average after filter:",
    round(
        total_after_filter / len(s1),
        2
    )
)

print("\nDone.")