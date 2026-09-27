from pathlib import Path
import csv
from collections import Counter

dataset_folder = Path("Dataset")

for file in sorted(dataset_folder.rglob("*.tsv")):

    # Skip ground truth
    if "ground_truth" in file.name:
        continue

    country_counts = Counter()

    with open(file, "r", encoding="utf-8", errors="replace", newline="") as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            country_counts[row["country"].strip()] += 1

    total = sum(country_counts.values())

    print("\n" + "=" * 70)
    print("FILE:", file)
    print("TOTAL ROWS:", f"{total:,}")

    print("\nCOUNTRY DISTRIBUTION:")

    for country, count in country_counts.most_common():
        percentage = count / total * 100
        print(f"{country}: {count:,} ({percentage:.2f}%)")