import pandas as pd


BASE = "Dataset"

S1_PATH = f"{BASE}/train/train_source1.tsv"
S2_PATH = f"{BASE}/train/train_source2.tsv"
S3_PATH = f"{BASE}/train/train_source3.tsv"
GT_PATH = f"{BASE}/train/train_ground_truth.tsv"


print("Loading data...")

s1 = pd.read_csv(S1_PATH, sep="\t", nrows=10000)
s2 = pd.read_csv(S2_PATH, sep="\t", nrows=10000)
s3 = pd.read_csv(S3_PATH, sep="\t", nrows=10000)

gt = pd.read_csv(GT_PATH, sep="\t")

sources = pd.concat([s2, s3], ignore_index=True)

s1_lookup = s1.set_index("entity_id")
source_lookup = sources.set_index("entity_id")

prototype_ids = set(sources["entity_id"])


print("\nFinding real matches with missing addresses...")

count = 0

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    if s1_id not in s1_lookup.index:
        continue

    matched_ids = str(
        row["matched_entity_ids"]
    ).split(",")

    for match_id in matched_ids:

        match_id = match_id.strip()

        if match_id not in prototype_ids:
            continue

        address1 = s1_lookup.loc[
            s1_id,
            "business_address"
        ]

        address2 = source_lookup.loc[
            match_id,
            "business_address"
        ]

        missing1 = pd.isna(address1)
        missing2 = pd.isna(address2)

        if missing1 or missing2:

            print("\n--------------------------------")
            print("S1:", s1_id)
            print("Match:", match_id)

            print(
                "S1 address:",
                address1
            )

            print(
                "Match address:",
                address2
            )

            print(
                "address_missing:",
                int(missing1 or missing2)
            )

            count += 1

        if count >= 10:
            break

    if count >= 10:
        break


print("\n--------------------------------")
print("Missing-address true matches shown:", count)
print("--------------------------------")