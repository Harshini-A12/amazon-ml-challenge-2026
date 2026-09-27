import pandas as pd
from rapidfuzz import fuzz


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

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    nrows=10000
)

s2 = pd.read_csv(
    S2_PATH,
    sep="\t",
    nrows=10000
)

s3 = pd.read_csv(
    S3_PATH,
    sep="\t",
    nrows=10000
)

gt = pd.read_csv(
    GT_PATH,
    sep="\t"
)


# ========================================
# COMBINE SOURCES
# ========================================

sources = pd.concat(
    [s2, s3],
    ignore_index=True
)

source_lookup = sources.set_index("entity_id")
s1_lookup = s1.set_index("entity_id")

prototype_ids = set(
    sources["entity_id"]
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

            pairs.append(
                (s1_id, match_id)
            )

        if len(pairs) >= 30:
            break

    if len(pairs) >= 30:
        break


print("Real match pairs found:", len(pairs))


# ========================================
# NORMALIZATION
# ========================================

def normalize(text):

    if pd.isna(text):
        return ""

    return str(text).lower().strip()


# ========================================
# ADDRESS SIMILARITY
# ========================================

print("\n========================================")
print("ADDRESS SIMILARITY EXPERIMENT")
print("========================================")


for s1_id, match_id in pairs:

    address1 = normalize(
        s1_lookup.loc[
            s1_id,
            "business_address"
        ]
    )

    address2 = normalize(
        source_lookup.loc[
            match_id,
            "business_address"
        ]
    )


    # Missing address
    if not address1 or not address2:

        print("\n----------------------------------------")

        print("S1:", s1_id)
        print("Match:", match_id)

        print("Address 1:", address1)
        print("Address 2:", address2)

        print("Address missing on one side.")

        continue


    wratio = fuzz.WRatio(
        address1,
        address2
    )

    token_sort = fuzz.token_sort_ratio(
        address1,
        address2
    )

    token_set = fuzz.token_set_ratio(
        address1,
        address2
    )


    print("\n----------------------------------------")

    print("S1:", s1_id)
    print("Match:", match_id)

    print("Address 1:", address1)
    print("Address 2:", address2)

    print(
        "WRatio:",
        round(wratio, 2)
    )

    print(
        "Token Sort:",
        round(token_sort, 2)
    )

    print(
        "Token Set:",
        round(token_set, 2)
    )


print("\nExperiment complete.")