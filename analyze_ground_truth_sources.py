from pathlib import Path
import csv

ground_truth_file = Path("Dataset/train/train_ground_truth.tsv")

source2_ids = set()
source3_ids = set()

# First collect all valid IDs from Source 2
with open(
    "Dataset/train/train_source2.tsv",
    "r",
    encoding="utf-8",
    errors="replace",
    newline=""
) as f:

    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        source2_ids.add(row["entity_id"])


# Then collect all valid IDs from Source 3
with open(
    "Dataset/train/train_source3.tsv",
    "r",
    encoding="utf-8",
    errors="replace",
    newline=""
) as f:

    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        source3_ids.add(row["entity_id"])


# Now analyze the ground truth
total_matches = 0
source2_matches = 0
source3_matches = 0
invalid_matches = 0

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

        if not matched_ids:
            continue

        for entity_id in matched_ids.split(","):

            entity_id = entity_id.strip()
            total_matches += 1

            if entity_id in source2_ids:
                source2_matches += 1

            elif entity_id in source3_ids:
                source3_matches += 1

            else:
                invalid_matches += 1


print("\n" + "=" * 70)
print("GROUND TRUTH SOURCE ANALYSIS")
print("=" * 70)

print("TOTAL MATCHED RECORDS:", f"{total_matches:,}")
print("SOURCE 2 MATCHES:", f"{source2_matches:,}")
print("SOURCE 3 MATCHES:", f"{source3_matches:,}")
print("INVALID MATCH IDs:", f"{invalid_matches:,}")

print("\nPERCENTAGE OF MATCHES:")

print(
    "Source 2:",
    f"{source2_matches / total_matches * 100:.2f}%"
)

print(
    "Source 3:",
    f"{source3_matches / total_matches * 100:.2f}%"
)