import pandas as pd
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

SAMPLE_S1 = 50000
NEGATIVES_PER_POSITIVE = 5

# ============================================================
# 1. LOAD GROUND TRUTH
# ============================================================

print("Loading ground truth...")

gt = pd.read_csv(
    "Dataset/train/train_ground_truth.tsv",
    sep="\t"
)

print("Ground truth rows:", len(gt))

# Keep only S1 entities that actually have matches
gt = gt[gt["matched_entity_ids"].notna()].copy()
gt = gt[gt["matched_entity_ids"].str.strip() != ""]

# ============================================================
# 2. SAMPLE S1 ENTITIES
# ============================================================

print("\nSampling S1 entities...")

sample = gt.sample(
    n=min(SAMPLE_S1, len(gt)),
    random_state=42
)

print("Sampled S1 entities:", len(sample))

# ============================================================
# 3. EXPLODE TRUE MATCHES
# ============================================================

print("\nExtracting positive pairs...")

positive_rows = []

for _, row in sample.iterrows():

    matched_ids = row["matched_entity_ids"].split(",")

    for match_id in matched_ids:

        positive_rows.append({
            "source1_entity_id": row["source1_entity_id"],
            "matched_entity_id": match_id.strip(),
            "label": 1
        })

positive_df = pd.DataFrame(positive_rows)

print("Positive pairs:", len(positive_df))

# ============================================================
# 4. SAVE POSITIVE PAIRS
# ============================================================

positive_df.to_csv(
    "training_positive_pairs.csv",
    index=False
)

print("\nSaved:")
print("training_positive_pairs.csv")

print("\nDONE.")