import pandas as pd
import re


GROUND_TRUTH = "Dataset/train/train_ground_truth.tsv"
SOURCE1 = "Dataset/train/train_source1.tsv"
SOURCE2 = "Dataset/train/train_source2.tsv"
SOURCE3 = "Dataset/train/train_source3.tsv"


def normalize_name(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Read only the required columns
gt = pd.read_csv(GROUND_TRUTH, sep="\t")

s1 = pd.read_csv(
    SOURCE1,
    sep="\t",
    usecols=["entity_id", "business_name"]
)

s2 = pd.read_csv(
    SOURCE2,
    sep="\t",
    usecols=["entity_id", "business_name"]
)

s3 = pd.read_csv(
    SOURCE3,
    sep="\t",
    usecols=["entity_id", "business_name"]
)


# Create lookups
s1_name = dict(zip(s1["entity_id"], s1["business_name"]))
s2_name = dict(zip(s2["entity_id"], s2["business_name"]))
s3_name = dict(zip(s3["entity_id"], s3["business_name"]))


same_first_char = 0
different_first_char = 0
checked_matches = 0


# Check first 1000 Source1 entities
for _, row in gt.head(1000).iterrows():

    source1_id = row["source1_entity_id"]
    matches = str(row["matched_entity_ids"])

    if matches == "nan" or matches.strip() == "":
        continue

    name1 = normalize_name(s1_name[source1_id])

    if not name1:
        continue

    first_char_1 = name1[0]

    for matched_id in matches.split(","):

        matched_id = matched_id.strip()

        if matched_id.startswith("S2-"):
            name2 = normalize_name(s2_name[matched_id])
        else:
            name2 = normalize_name(s3_name[matched_id])

        if not name2:
            continue

        first_char_2 = name2[0]

        checked_matches += 1

        if first_char_1 == first_char_2:
            same_first_char += 1
        else:
            different_first_char += 1


print("True matches checked:", checked_matches)
print("Same first character:", same_first_char)
print("Different first character:", different_first_char)

if checked_matches > 0:
    print(
        "First-character blocking recall:",
        round((same_first_char / checked_matches) * 100, 2),
        "%"
    )