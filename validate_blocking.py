import pandas as pd
import re
import ast
from collections import defaultdict

# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 10000

BASE = "Dataset"

S1_PATH = f"{BASE}/train/train_source1.tsv"
S2_PATH = f"{BASE}/train/train_source2.tsv"
S3_PATH = f"{BASE}/train/train_source3.tsv"
GT_PATH = f"{BASE}/train/train_ground_truth.tsv"


# ============================================================
# TOKENIZATION
# ============================================================

LEGAL_WORDS = {
    "inc", "incorporated", "llc", "ltd", "limited",
    "pvt", "private", "plc", "corp", "corporation",
    "co", "company", "llp"
}


def get_tokens(text):
    if pd.isna(text):
        return set()

    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()

    return {
        t for t in tokens
        if len(t) > 1 and t not in LEGAL_WORDS
    }


# ============================================================
# LOAD S1 SAMPLE
# ============================================================

print("Loading Source 1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    nrows=SAMPLE_SIZE
)

s1 = s1.reset_index(drop=True)

print("S1 rows:", len(s1))


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t"
)

gt = gt[gt["source1_entity_id"].isin(s1["entity_id"])]

print("Ground-truth S1 entities:", len(gt))


# ============================================================
# BUILD TRUE MATCH SET
# ============================================================

true_matches = defaultdict(set)

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]
    value = row["matched_entity_ids"]

    if pd.isna(value) or str(value).strip() == "":
        continue

    for entity_id in str(value).split(","):
        true_matches[s1_id].add(entity_id.strip())


total_true_matches = sum(
    len(v) for v in true_matches.values()
)

print("True matches:", total_true_matches)


# ============================================================
# BUILD S1 TOKEN MAPS
# ============================================================

print("\nPreparing S1 tokens...")

s1_name_tokens = {}
s1_address_tokens = {}
s1_country = {}

for _, row in s1.iterrows():

    sid = row["entity_id"]

    s1_name_tokens[sid] = get_tokens(row["business_name"])
    s1_address_tokens[sid] = get_tokens(row["business_address"])
    s1_country[sid] = str(row["country"]).lower().strip()


# ============================================================
# BUILD TOKEN → S1 INDEX
# ============================================================

print("Building S1 token index...")

name_index = defaultdict(set)
address_index = defaultdict(set)

for sid in s1["entity_id"]:

    for token in s1_name_tokens[sid]:
        name_index[token].add(sid)

    for token in s1_address_tokens[sid]:
        address_index[token].add(sid)

print("Name tokens:", len(name_index))
print("Address tokens:", len(address_index))


# ============================================================
# PROCESS SOURCE 2 + SOURCE 3
# ============================================================

found_matches = set()
total_candidate_pairs = 0

source_files = [S2_PATH, S3_PATH]

for source_number, path in enumerate(source_files, start=2):

    print(f"\nProcessing Source {source_number}...")

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=100000
    ):

        for _, row in chunk.iterrows():

            candidate_s1 = set()

            name_tokens = get_tokens(row["business_name"])
            address_tokens = get_tokens(row["business_address"])

            # Candidates through name
            for token in name_tokens:
                candidate_s1.update(name_index.get(token, ()))

            # Candidates through address
            for token in address_tokens:
                candidate_s1.update(address_index.get(token, ()))

            if not candidate_s1:
                continue

            country = str(row["country"]).lower().strip()
            candidate_id = row["entity_id"]

            for sid in candidate_s1:

                # Country blocking
                if s1_country[sid] != country:
                    continue

                name_overlap = len(
                    s1_name_tokens[sid] & name_tokens
                )

                address_overlap = len(
                    s1_address_tokens[sid] & address_tokens
                )

                # Our current second-stage filter
                if name_overlap >= 2 or address_overlap >= 2:

                    total_candidate_pairs += 1

                    if candidate_id in true_matches.get(sid, set()):
                        found_matches.add((sid, candidate_id))

        print(
            f"Processed chunk; true matches recovered so far: "
            f"{len(found_matches)}"
        )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("BLOCKING VALIDATION RESULTS")
print("=" * 60)

print("Total ground-truth matches:", total_true_matches)
print("Recovered ground-truth matches:", len(found_matches))

if total_true_matches > 0:

    recall = len(found_matches) / total_true_matches

    print(f"Blocking recall: {recall:.4%}")

print("Final candidate pairs generated:", total_candidate_pairs)

# Entity-level recall
entities_with_matches = {
    sid for sid, matches in true_matches.items()
    if matches
}

entities_recovered = {
    sid for sid, cid in found_matches
}

print(
    "Entities with at least one true match:",
    len(entities_with_matches)
)

print(
    "Entities with at least one recovered match:",
    len(entities_recovered)
)

if entities_with_matches:

    entity_recall = (
        len(entities_recovered)
        / len(entities_with_matches)
    )

    print(f"Entity recall: {entity_recall:.4%}")

print("\nDone.")