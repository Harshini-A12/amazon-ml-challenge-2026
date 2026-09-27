import pandas as pd
import re


# ========================================
# FILE PATHS
# ========================================

BASE = "Dataset"

S1_PATH = f"{BASE}/train/train_source1.tsv"
S2_PATH = f"{BASE}/train/train_source2.tsv"
S3_PATH = f"{BASE}/train/train_source3.tsv"
GT_PATH = f"{BASE}/train/train_ground_truth.tsv"


# ========================================
# NORMALIZATION
# ========================================

STOPWORDS = {
    "inc", "incorporated", "llc", "ltd", "limited",
    "corp", "corporation", "company", "co",
    "private", "pvt", "plc"
}


def normalize(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def useful_tokens(text):

    return {
        token
        for token in normalize(text).split()
        if token not in STOPWORDS and len(token) > 1
    }


def address_tokens(text):

    return {
        token
        for token in normalize(text).split()
        if len(token) > 1
    }


# ========================================
# LOAD DATA
# ========================================

print("Loading data...")

s1 = pd.read_csv(S1_PATH, sep="\t", nrows=10000)
s2 = pd.read_csv(S2_PATH, sep="\t", nrows=10000)
s3 = pd.read_csv(S3_PATH, sep="\t", nrows=10000)

gt = pd.read_csv(GT_PATH, sep="\t")

sources = pd.concat([s2, s3], ignore_index=True)

prototype_source_ids = set(sources["entity_id"])

print("S1:", len(s1))
print("Prototype S2 + S3:", len(sources))
print("Ground truth:", len(gt))


# ========================================
# CREATE INDEXES
# ========================================

country_index = {}
name_index = {}
address_index = {}


for _, row in sources.iterrows():

    entity_id = row["entity_id"]
    country = row["country"]

    # Country
    country_index.setdefault(country, set()).add(entity_id)

    # Name
    for token in useful_tokens(row["business_name"]):
        name_index.setdefault(token, set()).add(entity_id)

    # Address
    for token in address_tokens(row["business_address"]):
        address_index.setdefault(token, set()).add(entity_id)


# ========================================
# SOURCE1 LOOKUP
# ========================================

s1_lookup = s1.set_index("entity_id")


# ========================================
# FIND SUITABLE ENTITIES
# ========================================

suitable_entities = []

for _, row in gt.iterrows():

    matched_ids = str(row["matched_entity_ids"]).split(",")

    if any(
        x.strip() in prototype_source_ids
        for x in matched_ids
    ):

        s1_id = row["source1_entity_id"]

        if s1_id in s1_lookup.index:
            suitable_entities.append(s1_id)

    if len(suitable_entities) >= 500:
        break


print("\nSuitable entities:", len(suitable_entities))


# ========================================
# DIAGNOSTIC COUNTERS
# ========================================

name_only = []
address_only = []
both_routes = []
both_missed = []


# ========================================
# DIAGNOSTIC EVALUATION
# ========================================

for s1_id in suitable_entities:

    s1_row = s1_lookup.loc[s1_id]

    name = s1_row["business_name"]
    address = s1_row["business_address"]
    country = s1_row["country"]

    gt_row = gt[
        gt["source1_entity_id"] == s1_id
    ].iloc[0]

    true_matches = {
        x.strip()
        for x in str(
            gt_row["matched_entity_ids"]
        ).split(",")
        if x.strip()
        and x.strip() in prototype_source_ids
    }

    if not true_matches:
        continue


    # Country block
    country_candidates = country_index.get(
        country, set()
    )


    # Name route
    name_candidate_ids = set()

    for token in useful_tokens(name):

        name_candidate_ids.update(
            name_index.get(token, set())
        )

    name_candidates = (
        name_candidate_ids
        & country_candidates
    )


    # Address route
    address_candidate_ids = set()

    for token in address_tokens(address):

        address_candidate_ids.update(
            address_index.get(token, set())
        )

    address_candidates = (
        address_candidate_ids
        & country_candidates
    )


    # Check every true match
    for true_id in true_matches:

        in_name = true_id in name_candidates
        in_address = true_id in address_candidates


        if in_name and in_address:

            both_routes.append({
                "s1_id": s1_id,
                "true_id": true_id
            })


        elif in_name and not in_address:

            name_only.append({
                "s1_id": s1_id,
                "true_id": true_id,
                "name": name,
                "address": address
            })


        elif in_address and not in_name:

            address_only.append({
                "s1_id": s1_id,
                "true_id": true_id,
                "name": name,
                "address": address
            })


        else:

            both_missed.append({
                "s1_id": s1_id,
                "true_id": true_id,
                "name": name,
                "address": address
            })


# ========================================
# SUMMARY
# ========================================

print("\n========================================")
print("DIFFICULT CASE DIAGNOSTIC")
print("========================================")

print(
    "Caught by BOTH routes:",
    len(both_routes)
)

print(
    "Caught ONLY by NAME:",
    len(name_only)
)

print(
    "Caught ONLY by ADDRESS:",
    len(address_only)
)

print(
    "MISSED BY BOTH:",
    len(both_missed)
)


# ========================================
# SHOW NAME-ONLY CASES
# ========================================

print("\n========================================")
print("EXAMPLES: NAME ONLY")
print("========================================")

for case in name_only[:5]:

    print("\nS1:", case["s1_id"])
    print("True match:", case["true_id"])
    print("S1 name:", case["name"])
    print("S1 address:", case["address"])


# ========================================
# SHOW ADDRESS-ONLY CASES
# ========================================

print("\n========================================")
print("EXAMPLES: ADDRESS ONLY")
print("========================================")

for case in address_only[:5]:

    print("\nS1:", case["s1_id"])
    print("True match:", case["true_id"])
    print("S1 name:", case["name"])
    print("S1 address:", case["address"])


# ========================================
# SHOW BOTH-MISSED CASES
# ========================================

print("\n========================================")
print("EXAMPLES: MISSED BY BOTH")
print("========================================")

if both_missed:

    for case in both_missed[:10]:

        print("\nS1:", case["s1_id"])
        print("True match:", case["true_id"])
        print("S1 name:", case["name"])
        print("S1 address:", case["address"])

else:

    print("NONE")


print("\nDiagnostic complete.")