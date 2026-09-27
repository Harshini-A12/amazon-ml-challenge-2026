from pathlib import Path
import csv

dataset_folder = Path("Dataset")

for file in sorted(dataset_folder.rglob("*.tsv")):

    if "ground_truth" in file.name:
        continue

    entity_ids = set()
    duplicate_ids = 0

    with open(file, "r", encoding="utf-8", errors="replace", newline="") as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            entity_id = row["entity_id"]

            if entity_id in entity_ids:
                duplicate_ids += 1
            else:
                entity_ids.add(entity_id)

    print("\n" + "=" * 70)
    print("FILE:", file)
    print("TOTAL UNIQUE ENTITY IDs:", f"{len(entity_ids):,}")
    print("DUPLICATE ENTITY ID ROWS:", f"{duplicate_ids:,}")