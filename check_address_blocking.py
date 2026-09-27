import pandas as pd
import re


GROUND_TRUTH = "Dataset/train/train_ground_truth.tsv"
SOURCE1 = "Dataset/train/train_source1.tsv"
SOURCE2 = "Dataset/train/train_source2.tsv"
SOURCE3 = "Dataset/train/train_source3.tsv"


def normalize_address(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


gt = pd.read_csv(GROUND_TRUTH, sep="\t")

s1 = pd.read_csv(
    SOURCE1, sep="\t",
    usecols=["entity_id", "business_address"]
)

s2 = pd.read_csv(
    SOURCE2, sep="\t",
    usecols=["entity_id", "business_address"]
)

s3 = pd.read_csv(
    SOURCE3, sep="\t",
    usecols=["entity_id", "business_address"]
)


s1_address = dict(zip(s1["entity_id"], s1["business_address"]))
s2_address = dict(zip(s2["entity_id"], s2["business_address"]))
s3_address = dict(zip(s3["entity_id"], s3["business_address"]))


same_first_token = 0
different_first_token = 0
no_address = 0
checked_matches = 0


for _, row in gt.head(1000).iterrows():

    source1_id = row["source1_entity_id"]
    matches = str(row["matched_entity_ids"])

    if matches == "nan" or matches.strip() == "":
        continue

    address1 = normalize_address(s1_address[source1_id])

    if not address1:
        continue

    tokens1 = address1.split()

    if not tokens1:
        continue

    first_token_1 = tokens1[0]

    for matched_id in matches.split(","):

        matched_id = matched_id.strip()

        if matched_id.startswith("S2-"):
            address2 = normalize_address(s2_address[matched_id])
        else:
            address2 = normalize_address(s3_address[matched_id])

        if not address2:
            no_address += 1
            continue

        tokens2 = address2.split()

        if not tokens2:
            continue

        first_token_2 = tokens2[0]

        checked_matches += 1

        if first_token_1 == first_token_2:
            same_first_token += 1
        else:
            different_first_token += 1


print("True matches checked:", checked_matches)
print("Same first address token:", same_first_token)
print("Different first address token:", different_first_token)
print("Missing candidate address:", no_address)

if checked_matches > 0:
    print(
        "First-address-token blocking recall:",
        round((same_first_token / checked_matches) * 100, 2),
        "%"
    )