import pandas as pd
import re
import unicodedata
from collections import defaultdict

S1_FILE = "Dataset/test/test_source1.tsv"
S2_FILE = "Dataset/test/test_source2.tsv"
S3_FILE = "Dataset/test/test_source3.tsv"

OUTPUT_FILE = "submission.tsv"

CHUNK = 100000


# ============================================================
# SAME NORMALIZATION AS TRAINING
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = "".join(
        " " if unicodedata.category(c).startswith("P") else c
        for c in text
    )

    return re.sub(r"\s+", " ", text).strip()


def key(text):
    return normalize_text(text)


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("=" * 60)
print("LOADING SOURCE 1")
print("=" * 60)

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ],
    dtype=str
)

print("S1 rows:", len(s1))


# ============================================================
# BUILD COMPACT LOOKUPS
# ============================================================

print("\nBuilding name lookup...")

name_index = defaultdict(list)
address_index = defaultdict(list)

for row in s1.itertuples(index=False):

    entity_id = row.entity_id
    country = str(row.country).lower().strip()

    name = key(row.business_name)
    address = key(row.business_address)

    if name:
        name_index[(country, name)].append(entity_id)

    if address:
        address_index[(country, address)].append(entity_id)


print("Name keys:", len(name_index))
print("Address keys:", len(address_index))


# ============================================================
# STORE MATCHES ONLY
# ============================================================

matches = defaultdict(set)


# ============================================================
# STREAM ONE SOURCE
# ============================================================

def process_source(filename, label):

    print("\n" + "=" * 60)
    print("PROCESSING", label)
    print("=" * 60)

    total = 0
    matched_rows = 0

    for chunk in pd.read_csv(
        filename,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ],
        dtype=str,
        chunksize=CHUNK
    ):

        for row in chunk.itertuples(index=False):

            total += 1

            country = str(row.country).lower().strip()

            name = key(row.business_name)
            address = key(row.business_address)

            found = set()

            # Exact normalized name
            if name:
                found.update(
                    name_index.get(
                        (country, name),
                        []
                    )
                )

            # Exact normalized address
            if address:
                found.update(
                    address_index.get(
                        (country, address),
                        []
                    )
                )

            if found:

                matched_rows += 1

                for s1_id in found:
                    matches[s1_id].add(row.entity_id)

        print(
            f"Processed {total:,} | "
            f"matched source rows {matched_rows:,}"
        )

    print(
        label,
        "complete:",
        total,
        "rows"
    )


# ============================================================
# PROCESS S2 + S3
# ============================================================

process_source(S2_FILE, "SOURCE 2")
process_source(S3_FILE, "SOURCE 3")


# ============================================================
# WRITE FINAL SUBMISSION
# ============================================================

print("\n" + "=" * 60)
print("WRITING SUBMISSION")
print("=" * 60)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    f.write(
        "source1_entity_id\tmatched_entity_ids\n"
    )

    for entity_id in s1["entity_id"]:

        ids = matches.get(entity_id, set())

        f.write(
            str(entity_id)
            + "\t"
            + ",".join(sorted(ids))
            + "\n"
        )


print("\n" + "=" * 60)
print("SUBMISSION COMPLETE")
print("=" * 60)

print("S1 rows:", len(s1))
print("S1 entities with matches:", len(matches))
print("File:", OUTPUT_FILE)