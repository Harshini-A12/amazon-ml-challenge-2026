import pandas as pd

GROUND_TRUTH = "Dataset/Train/train_ground_truth.tsv"

# Take only the first 500 true matches
ground_truth = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    nrows=500
)

matched_ids = []

for _, row in ground_truth.iterrows():

    if pd.isna(row["matched_entity_ids"]) or row["matched_entity_ids"] == "":
        continue

    # Take the first matched entity for each Source 1
    matched_ids.append(row["matched_entity_ids"].split(",")[0])

    if len(matched_ids) == 100:
        break


source2_ids = {
    x for x in matched_ids
    if x.startswith("S2-")
}

source3_ids = {
    x for x in matched_ids
    if x.startswith("S3-")
}


print("True matches being checked:")
print("Source 2:", len(source2_ids))
print("Source 3:", len(source3_ids))


# --------------------------------------------------
# Check Source 2
# --------------------------------------------------

s2_found = pd.DataFrame()

if source2_ids:

    for chunk in pd.read_csv(
        "Dataset/Train/train_source2.tsv",
        sep="\t",
        chunksize=100000
    ):

        matches = chunk[
            chunk["entity_id"].isin(source2_ids)
        ]

        if len(matches) > 0:
            s2_found = pd.concat(
                [s2_found, matches]
            )

        if len(s2_found) >= len(source2_ids):
            break


# --------------------------------------------------
# Check Source 3
# --------------------------------------------------

s3_found = pd.DataFrame()

if source3_ids:

    for chunk in pd.read_csv(
        "Dataset/Train/train_source3.tsv",
        sep="\t",
        chunksize=100000
    ):

        matches = chunk[
            chunk["entity_id"].isin(source3_ids)
        ]

        if len(matches) > 0:
            s3_found = pd.concat(
                [s3_found, matches]
            )

        if len(s3_found) >= len(source3_ids):
            break


# --------------------------------------------------
# Results
# --------------------------------------------------

s2_missing = s2_found["business_address"].isna().sum()
s3_missing = s3_found["business_address"].isna().sum()

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(
    "Source 2 matched records checked:",
    len(s2_found)
)

print(
    "Source 2 with missing address:",
    s2_missing
)

if len(s2_found) > 0:
    print(
        "Source 2 missing percentage:",
        round(s2_missing / len(s2_found) * 100, 2),
        "%"
    )


print()

print(
    "Source 3 matched records checked:",
    len(s3_found)
)

print(
    "Source 3 with missing address:",
    s3_missing
)

if len(s3_found) > 0:
    print(
        "Source 3 missing percentage:",
        round(s3_missing / len(s3_found) * 100, 2),
        "%"
    )