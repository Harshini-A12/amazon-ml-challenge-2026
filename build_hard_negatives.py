import pandas as pd
import re
import unicodedata

# ============================================================
# SETTINGS
# ============================================================

POSITIVE_FILE = "training_positive_pairs.csv"

S1_FILE = "Dataset/train/train_source1.tsv"
S2_FILE = "Dataset/train/train_source2.tsv"
S3_FILE = "Dataset/train/train_source3.tsv"

OUTPUT_FILE = "training_hard_negatives.csv"

CHUNK_SIZE = 100000
NEGATIVES_PER_ENTITY = 3


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
# 1. LOAD POSITIVE PAIRS
# ============================================================

print("Loading positive pairs...")

positive_df = pd.read_csv(POSITIVE_FILE)

print("Positive pairs:", len(positive_df))


# ============================================================
# 2. TRUE MATCH LOOKUP
# ============================================================

true_matches = {}

for row in positive_df.itertuples(index=False):

    s1_id = row.source1_entity_id
    match_id = row.matched_entity_id

    if s1_id not in true_matches:
        true_matches[s1_id] = set()

    true_matches[s1_id].add(match_id)

print("S1 entities:", len(true_matches))


# ============================================================
# 3. LOAD ONLY OUR 50K S1 RECORDS
# ============================================================

print("\nLoading S1...")

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

s1 = s1[s1["entity_id"].isin(true_matches)].copy()

print("S1 training entities:", len(s1))


# ============================================================
# 4. BUILD INDEX FROM S1 ONLY
# ============================================================

print("\nBuilding lightweight S1 token index...")

token_to_s1 = {}

s1_country = {}

for row in s1.itertuples(index=False):

    s1_id = row.entity_id

    s1_country[s1_id] = row.country

    tokens = (
        get_tokens(row.business_name)
        |
        get_tokens(row.business_address)
    )

    for token in tokens:

        if token not in token_to_s1:
            token_to_s1[token] = set()

        token_to_s1[token].add(s1_id)

print("Unique S1 tokens:", len(token_to_s1))


# ============================================================
# 5. TRACK NEGATIVES
# ============================================================

negative_count = {
    s1_id: 0
    for s1_id in true_matches
}

negative_pairs = set()

negative_rows = []


# ============================================================
# 6. PROCESS SOURCE FILE
# ============================================================

def process_source(source_file):

    print("\nProcessing:", source_file)

    total_rows = 0

    for chunk in pd.read_csv(
        source_file,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ],
        chunksize=CHUNK_SIZE
    ):

        for row in chunk.itertuples(index=False):

            source_id = row.entity_id
            country = row.country

            tokens = (
                get_tokens(row.business_name)
                |
                get_tokens(row.business_address)
            )

            candidate_s1 = set()

            for token in tokens:

                if token in token_to_s1:
                    candidate_s1.update(
                        token_to_s1[token]
                    )

            for s1_id in candidate_s1:

                # Already enough negatives
                if negative_count[s1_id] >= NEGATIVES_PER_ENTITY:
                    continue

                # Country must match
                if s1_country[s1_id] != country:
                    continue

                # Must NOT be a true match
                if source_id in true_matches[s1_id]:
                    continue

                pair = (s1_id, source_id)

                # Avoid duplicate negative pair
                if pair in negative_pairs:
                    continue

                negative_pairs.add(pair)

                negative_rows.append({
                    "source1_entity_id": s1_id,
                    "matched_entity_id": source_id,
                    "label": 0
                })

                negative_count[s1_id] += 1

            total_rows += 1

        print(
            "Processed rows:",
            total_rows,
            "| Negatives:",
            len(negative_rows)
        )

        # If every S1 entity has enough negatives,
        # we can stop reading the remaining source data.
        if all(
            count >= NEGATIVES_PER_ENTITY
            for count in negative_count.values()
        ):
            print("Enough negatives found for every S1 entity.")
            return True

    return False


# ============================================================
# 7. STREAM SOURCE 2
# ============================================================

finished = process_source(S2_FILE)


# ============================================================
# 8. STREAM SOURCE 3 IF NEEDED
# ============================================================

if not finished:

    print("\nSource 2 wasn't enough.")
    print("Continuing with Source 3...")

    process_source(S3_FILE)


# ============================================================
# 9. SAVE
# ============================================================

negative_df = pd.DataFrame(negative_rows)

print("\n============================================")
print("HARD NEGATIVE GENERATION COMPLETE")
print("============================================")

print("Hard negatives:", len(negative_df))

print(
    "S1 entities with 3 negatives:",
    sum(
        count >= NEGATIVES_PER_ENTITY
        for count in negative_count.values()
    )
)

print(
    "S1 entities with fewer than 3:",
    sum(
        count < NEGATIVES_PER_ENTITY
        for count in negative_count.values()
    )
)

negative_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:", OUTPUT_FILE)