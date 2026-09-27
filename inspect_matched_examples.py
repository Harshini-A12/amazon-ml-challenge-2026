from pathlib import Path
import csv

ground_truth_file = Path("Dataset/train/train_ground_truth.tsv")

source1_file = Path("Dataset/train/train_source1.tsv")
source2_file = Path("Dataset/train/train_source2.tsv")
source3_file = Path("Dataset/train/train_source3.tsv")


# --------------------------------------------------
# STEP 1: Pick a few Source 1 entities with matches
# --------------------------------------------------

examples = []

with open(
    ground_truth_file,
    "r",
    encoding="utf-8",
    errors="replace",
    newline=""
) as f:

    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:

        matched_ids = row["matched_entity_ids"].strip()

        if matched_ids:
            examples.append(
                (
                    row["source1_entity_id"],
                    matched_ids.split(",")
                )
            )

        if len(examples) == 5:
            break


# --------------------------------------------------
# STEP 2: Load only the records we need
# --------------------------------------------------

needed_s1 = {example[0] for example in examples}

needed_s2 = set()
needed_s3 = set()

for _, matched_ids in examples:

    for entity_id in matched_ids:

        if entity_id.startswith("S2-"):
            needed_s2.add(entity_id)

        elif entity_id.startswith("S3-"):
            needed_s3.add(entity_id)


def get_records(file_path, needed_ids):

    records = {}

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            if row["entity_id"] in needed_ids:
                records[row["entity_id"]] = row

                if len(records) == len(needed_ids):
                    break

    return records


source1_records = get_records(source1_file, needed_s1)
source2_records = get_records(source2_file, needed_s2)
source3_records = get_records(source3_file, needed_s3)


# --------------------------------------------------
# STEP 3: Display the known matches
# --------------------------------------------------

print("\n" + "=" * 80)
print("KNOWN MATCHED RECORD EXAMPLES")
print("=" * 80)

for source1_id, matched_ids in examples:

    s1 = source1_records[source1_id]

    print("\n" + "-" * 80)
    print("SOURCE 1")
    print("ID      :", s1["entity_id"])
    print("NAME    :", s1["business_name"])
    print("ADDRESS :", s1["business_address"])
    print("COUNTRY :", s1["country"])

    print("\nKNOWN MATCHES:")

    for matched_id in matched_ids:

        if matched_id.startswith("S2-"):
            record = source2_records.get(matched_id)
            source = "SOURCE 2"

        else:
            record = source3_records.get(matched_id)
            source = "SOURCE 3"

        print("\n", source)
        print("ID      :", record["entity_id"])
        print("NAME    :", record["business_name"])
        print("ADDRESS :", record["business_address"])
        print("COUNTRY :", record["country"])