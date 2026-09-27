import pandas as pd
import re
import unicodedata


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

sources = pd.concat([s2, s3], ignore_index=True)

s1_lookup = s1.set_index("entity_id")
source_lookup = sources.set_index("entity_id")

prototype_ids = set(sources["entity_id"])


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


# ============================================================
# 4. BUILD BLOCKING INDEXES
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
# 5. GROUND TRUTH
# ============================================================

ground_truth = {}

for _, row in gt.iterrows():

    matched = str(row["matched_entity_ids"])

    if matched == "nan" or matched == "":

        ground_truth[row["source1_entity_id"]] = set()

    else:

        ground_truth[row["source1_entity_id"]] = {
            x.strip()
            for x in matched.split(",")
            if x.strip()
        }


# ============================================================
# 6. TEST FILTERS
# ============================================================

filter_results = {

    "name>=1 OR address>=1": {
        "found": 0,
        "total": 0
    },

    "name>=2 OR address>=2": {
        "found": 0,
        "total": 0
    },

    "name>=1 OR address>=2": {
        "found": 0,
        "total": 0
    },

    "name>=2 OR address>=1": {
        "found": 0,
        "total": 0
    },

    "name>=1 AND address>=1": {
        "found": 0,
        "total": 0
    }
}


# ============================================================
# 7. TEST CANDIDATES
# ============================================================

print("\nTesting cheap filters...")

for count, (_, row) in enumerate(
    s1.iterrows(),
    start=1
):

    s1_id = row["entity_id"]

    true_matches = (
        ground_truth.get(s1_id, set())
        & prototype_ids
    )

    # We only evaluate S1 entities that have
    # at least one true match in our prototype sources.
    if not true_matches:
        continue


    country = row["country"]

    s1_name_tokens = name_tokens(
        row["business_name"]
    )

    s1_address_tokens = address_tokens(
        row["business_address"]
    )


    # ----------------------------------------
    # Generate original union candidates
    # ----------------------------------------

    candidates = set()

    for token in s1_name_tokens:

        key = (country, token)

        if key in name_index:
            candidates.update(
                name_index[key]
            )

    for token in s1_address_tokens:

        key = (country, token)

        if key in address_index:
            candidates.update(
                address_index[key]
            )


    # ----------------------------------------
    # Test every true match
    # ----------------------------------------

    for match_id in true_matches:

        candidate_idx = sources.index[
            sources["entity_id"] == match_id
        ]

        if len(candidate_idx) == 0:
            continue

        candidate = sources.loc[
            candidate_idx[0]
        ]


        candidate_name_tokens = name_tokens(
            candidate["business_name"]
        )

        candidate_address_tokens = address_tokens(
            candidate["business_address"]
        )


        shared_name = len(
            s1_name_tokens &
            candidate_name_tokens
        )

        shared_address = len(
            s1_address_tokens &
            candidate_address_tokens
        )


        # Original blocking must contain it
        if candidate_idx[0] not in candidates:
            continue


        # ----------------------------------------
        # Filter 1
        # ----------------------------------------

        filter_results[
            "name>=1 OR address>=1"
        ]["total"] += 1

        if (
            shared_name >= 1
            or
            shared_address >= 1
        ):
            filter_results[
                "name>=1 OR address>=1"
            ]["found"] += 1


        # ----------------------------------------
        # Filter 2
        # ----------------------------------------

        filter_results[
            "name>=2 OR address>=2"
        ]["total"] += 1

        if (
            shared_name >= 2
            or
            shared_address >= 2
        ):
            filter_results[
                "name>=2 OR address>=2"
            ]["found"] += 1


        # ----------------------------------------
        # Filter 3
        # ----------------------------------------

        filter_results[
            "name>=1 OR address>=2"
        ]["total"] += 1

        if (
            shared_name >= 1
            or
            shared_address >= 2
        ):
            filter_results[
                "name>=1 OR address>=2"
            ]["found"] += 1


        # ----------------------------------------
        # Filter 4
        # ----------------------------------------

        filter_results[
            "name>=2 OR address>=1"
        ]["total"] += 1

        if (
            shared_name >= 2
            or
            shared_address >= 1
        ):
            filter_results[
                "name>=2 OR address>=1"
            ]["found"] += 1


        # ----------------------------------------
        # Filter 5
        # ----------------------------------------

        filter_results[
            "name>=1 AND address>=1"
        ]["total"] += 1

        if (
            shared_name >= 1
            and
            shared_address >= 1
        ):
            filter_results[
                "name>=1 AND address>=1"
            ]["found"] += 1


    if count % 1000 == 0:

        print(
            f"Processed {count}/10000"
        )


# ============================================================
# 8. RESULTS
# ============================================================

print("\n========================================")
print("CHEAP FILTER RECALL")
print("========================================")

for name, result in filter_results.items():

    total = result["total"]
    found = result["found"]

    recall = (
        found / total * 100
        if total > 0
        else 0
    )

    print(
        f"{name:<30} "
        f"{found}/{total} "
        f"({recall:.2f}%)"
    )

print("\nDone.")