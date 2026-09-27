import pandas as pd


# ========================================
# FILE PATHS
# ========================================

BASE = "Dataset"

S1_PATH = f"{BASE}/train/train_source1.tsv"
S2_PATH = f"{BASE}/train/train_source2.tsv"
S3_PATH = f"{BASE}/train/train_source3.tsv"
GT_PATH = f"{BASE}/train/train_ground_truth.tsv"


# ========================================
# LOAD DATA
# ========================================

print("Loading data...")

s1 = pd.read_csv(S1_PATH, sep="\t", nrows=10000)
s2 = pd.read_csv(S2_PATH, sep="\t", nrows=10000)
s3 = pd.read_csv(S3_PATH, sep="\t", nrows=10000)
gt = pd.read_csv(GT_PATH, sep="\t")

sources = pd.concat([s2, s3], ignore_index=True)

s1_lookup = s1.set_index("entity_id")
source_lookup = sources.set_index("entity_id")

prototype_ids = set(sources["entity_id"])


# ========================================
# TOKEN FUNCTIONS
# ========================================

STOPWORDS = {
    "inc", "incorporated", "llc", "ltd", "limited",
    "corp", "corporation", "company", "co",
    "private", "pvt", "plc"
}


def normalize(text):

    if pd.isna(text):
        return ""

    return (
        str(text)
        .lower()
        .replace(",", " ")
        .replace(".", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("/", " ")
        .replace("-", " ")
    )


def name_tokens(text):

    return {
        token
        for token in normalize(text).split()
        if token not in STOPWORDS
        and len(token) > 1
    }


def address_tokens(text):

    return {
        token
        for token in normalize(text).split()
        if len(token) > 1
    }


def overlap(tokens1, tokens2):

    if not tokens1 or not tokens2:
        return 0.0

    common = tokens1 & tokens2

    return len(common) / min(
        len(tokens1),
        len(tokens2)
    )


# ========================================
# FIND REAL MATCH PAIRS
# ========================================

pairs = []

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    if s1_id not in s1_lookup.index:
        continue

    matched_ids = str(
        row["matched_entity_ids"]
    ).split(",")

    for match_id in matched_ids:

        match_id = match_id.strip()

        if match_id in prototype_ids:

            pairs.append((s1_id, match_id))

        if len(pairs) >= 30:
            break

    if len(pairs) >= 30:
        break


print("Real match pairs found:", len(pairs))


# ========================================
# CALCULATE OVERLAP
# ========================================

print("\n========================================")
print("TOKEN OVERLAP EXPERIMENT")
print("========================================")


for s1_id, match_id in pairs:

    name1 = name_tokens(
        s1_lookup.loc[
            s1_id,
            "business_name"
        ]
    )

    name2 = name_tokens(
        source_lookup.loc[
            match_id,
            "business_name"
        ]
    )

    address1 = address_tokens(
        s1_lookup.loc[
            s1_id,
            "business_address"
        ]
    )

    address2 = address_tokens(
        source_lookup.loc[
            match_id,
            "business_address"
        ]
    )


    name_score = overlap(
        name1,
        name2
    )

    address_score = overlap(
        address1,
        address2
    )


    print("\n----------------------------------------")

    print("S1:", s1_id)
    print("Match:", match_id)

    print(
        "Name tokens:",
        name1
    )

    print(
        "Match name tokens:",
        name2
    )

    print(
        "Name overlap:",
        round(name_score, 2)
    )

    print(
        "Address overlap:",
        round(address_score, 2)
    )


print("\nExperiment complete.")