import pandas as pd

GROUND_TRUTH = "Dataset/train/train_ground_truth.tsv"
SOURCE1 = "Dataset/train/train_source1.tsv"
SOURCE2 = "Dataset/train/train_source2.tsv"
SOURCE3 = "Dataset/train/train_source3.tsv"


# Read only the columns we need
gt = pd.read_csv(GROUND_TRUTH, sep="\t")
s1 = pd.read_csv(SOURCE1, sep="\t", usecols=["entity_id", "country"])
s2 = pd.read_csv(SOURCE2, sep="\t", usecols=["entity_id", "country"])
s3 = pd.read_csv(SOURCE3, sep="\t", usecols=["entity_id", "country"])


# Create country lookup dictionaries
s1_country = dict(zip(s1["entity_id"], s1["country"]))
s2_country = dict(zip(s2["entity_id"], s2["country"]))
s3_country = dict(zip(s3["entity_id"], s3["country"]))


same_country = 0
different_country = 0
checked_matches = 0


# Check the first 1000 Source1 entities
for _, row in gt.head(1000).iterrows():

    source1_id = row["source1_entity_id"]
    matches = str(row["matched_entity_ids"])

    if matches == "nan" or matches.strip() == "":
        continue

    source1_country_value = s1_country[source1_id]

    for matched_id in matches.split(","):

        matched_id = matched_id.strip()

        if matched_id.startswith("S2-"):
            matched_country = s2_country[matched_id]
        else:
            matched_country = s3_country[matched_id]

        checked_matches += 1

        if source1_country_value == matched_country:
            same_country += 1
        else:
            different_country += 1


print("True matches checked:", checked_matches)
print("Same country:", same_country)
print("Different country:", different_country)

if checked_matches > 0:
    print(
        "Same-country percentage:",
        round((same_country / checked_matches) * 100, 2),
        "%"
    )