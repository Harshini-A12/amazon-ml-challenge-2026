import pandas as pd
import re
import unicodedata


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()

    # Replace punctuation with spaces while preserving all languages
    text = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    )

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# 1. Get 5 guaranteed true matches from ground truth
# --------------------------------------------------

ground_truth = pd.read_csv(
    "Dataset/Train/train_ground_truth.tsv",
    sep="\t",
    nrows=1000
)

examples = []

for _, row in ground_truth.iterrows():

    if pd.isna(row["matched_entity_ids"]) or row["matched_entity_ids"] == "":
        continue

    matched_ids = row["matched_entity_ids"].split(",")

    examples.append(
        (row["source1_entity_id"], matched_ids[0])
    )

    if len(examples) == 5:
        break


print("Selected guaranteed matches:")
for s1_id, match_id in examples:
    print(s1_id, "->", match_id)

print()


# --------------------------------------------------
# 2. Collect the exact IDs we need
# --------------------------------------------------

source1_ids = {s1_id for s1_id, _ in examples}
source2_ids = {
    match_id for _, match_id in examples
    if match_id.startswith("S2-")
}
source3_ids = {
    match_id for _, match_id in examples
    if match_id.startswith("S3-")
}


# --------------------------------------------------
# 3. Find exact records in Source 1
# --------------------------------------------------

source1_records = {}

for chunk in pd.read_csv(
    "Dataset/Train/train_source1.tsv",
    sep="\t",
    chunksize=100000
):
    matches = chunk[chunk["entity_id"].isin(source1_ids)]

    for _, row in matches.iterrows():
        source1_records[row["entity_id"]] = row

    if len(source1_records) == len(source1_ids):
        break


# --------------------------------------------------
# 4. Find exact records in Source 2
# --------------------------------------------------

source2_records = {}

for chunk in pd.read_csv(
    "Dataset/Train/train_source2.tsv",
    sep="\t",
    chunksize=100000
):
    matches = chunk[chunk["entity_id"].isin(source2_ids)]

    for _, row in matches.iterrows():
        source2_records[row["entity_id"]] = row

    if len(source2_records) == len(source2_ids):
        break


# --------------------------------------------------
# 5. Find exact records in Source 3
# --------------------------------------------------

source3_records = {}

for chunk in pd.read_csv(
    "Dataset/Train/train_source3.tsv",
    sep="\t",
    chunksize=100000
):
    matches = chunk[chunk["entity_id"].isin(source3_ids)]

    for _, row in matches.iterrows():
        source3_records[row["entity_id"]] = row

    if len(source3_records) == len(source3_ids):
        break


# --------------------------------------------------
# 6. Display original + normalized values
# --------------------------------------------------

print("=" * 80)

for s1_id, match_id in examples:

    s1 = source1_records[s1_id]

    if match_id.startswith("S2-"):
        matched = source2_records[match_id]
        source_name = "SOURCE 2"
    else:
        matched = source3_records[match_id]
        source_name = "SOURCE 3"

    print("\nSOURCE 1")
    print("ID:", s1["entity_id"])
    print("Name:", s1["business_name"])
    print("Address:", s1["business_address"])

    print("\n" + source_name)
    print("ID:", matched["entity_id"])
    print("Name:", matched["business_name"])
    print("Address:", matched["business_address"])

    print("\nNORMALIZED")
    print("S1 name :", normalize_text(s1["business_name"]))
    print("Match name:", normalize_text(matched["business_name"]))
    print("S1 addr :", normalize_text(s1["business_address"]))
    print("Match addr:", normalize_text(matched["business_address"]))

    print("\n" + "=" * 80)