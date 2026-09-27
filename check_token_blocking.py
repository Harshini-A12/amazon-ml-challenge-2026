import pandas as pd
import re


GROUND_TRUTH = "Dataset/train/train_ground_truth.tsv"
SOURCE1 = "Dataset/train/train_source1.tsv"
SOURCE2 = "Dataset/train/train_source2.tsv"
SOURCE3 = "Dataset/train/train_source3.tsv"


# Words that are usually not useful for identifying the business itself
STOPWORDS = {
    "inc", "incorporated", "llc", "ltd", "limited",
    "corp", "corporation", "company", "co",
    "private", "pvt", "plc"
}


def normalize_name(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def useful_tokens(text):
    normalized = normalize_name(text)

    return {
        token
        for token in normalized.split()
        if token not in STOPWORDS and len(token) > 1
    }


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


s1_name = dict(zip(s1["entity_id"], s1["business_name"]))
s2_name = dict(zip(s2["entity_id"], s2["business_name"]))
s3_name = dict(zip(s3["entity_id"], s3["business_name"]))


checked = 0
overlap_1 = 0
overlap_2 = 0
overlap_3 = 0


for _, row in gt.head(1000).iterrows():

    source1_id = row["source1_entity_id"]
    matches = str(row["matched_entity_ids"])

    if matches == "nan" or matches.strip() == "":
        continue

    tokens1 = useful_tokens(s1_name[source1_id])

    if not tokens1:
        continue

    for matched_id in matches.split(","):

        matched_id = matched_id.strip()

        if matched_id.startswith("S2-"):
            tokens2 = useful_tokens(s2_name[matched_id])
        else:
            tokens2 = useful_tokens(s3_name[matched_id])

        if not tokens2:
            continue

        overlap = len(tokens1 & tokens2)

        checked += 1

        if overlap >= 1:
            overlap_1 += 1

        if overlap >= 2:
            overlap_2 += 1

        if overlap >= 3:
            overlap_3 += 1


print("True matches checked:", checked)

if checked > 0:
    print(
        "At least 1 useful token:",
        overlap_1,
        "(",
        round(overlap_1 / checked * 100, 2),
        "%)"
    )

    print(
        "At least 2 useful tokens:",
        overlap_2,
        "(",
        round(overlap_2 / checked * 100, 2),
        "%)"
    )

    print(
        "At least 3 useful tokens:",
        overlap_3,
        "(",
        round(overlap_3 / checked * 100, 2),
        "%)"
    )
    