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
# LOAD SMALL PROTOTYPE
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


# Combine Source2 + Source3
sources = pd.concat(
    [s2, s3],
    ignore_index=True
)

source_lookup = sources.set_index("entity_id")

s1_lookup = s1.set_index("entity_id")


# ========================================
# FIND REAL MATCH PAIRS
# ========================================

pairs = []

prototype_ids = set(sources["entity_id"])

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
# CALCULATE SIMILARITIES
# ========================================

print("\n========================================")
print("NAME SIMILARITY EXPERIMENT")
print("========================================")


for s1_id, match_id in pairs:

    name1 = normalize(
        s1_lookup.loc[s1_id, "business_name"]
    )

    name2 = normalize(
        source_lookup.loc[match_id, "business_name"]
    )

    wratio = fuzz.WRatio(
        name1,
        name2
    )

    token_sort = fuzz.token_sort_ratio(
        name1,
        name2
    )

    token_set = fuzz.token_set_ratio(
        name1,
        name2
    )

    print("\n----------------------------------------")

    print("S1:", s1_id)
    print("Match:", match_id)

    print("Name 1:", name1)
    print("Name 2:", name2)

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