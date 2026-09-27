import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD
# ============================================================

print("Loading feature dataset...")

df = pd.read_csv(
    "prototype_features_v2.csv"
)

print("Total candidate pairs:", len(df))


# ============================================================
# 2. BASIC COUNTS
# ============================================================

print("\nOverall class distribution:")
print(df["label"].value_counts())

print(
    "\nTotal positive pairs:",
    df["label"].sum()
)


# ============================================================
# 3. COUNT S1 ENTITIES
# ============================================================

entities = df["source1_entity_id"].unique().tolist()

print(
    "\nUnique S1 entities:",
    len(entities)
)


# ============================================================
# 4. WHICH S1 ENTITIES HAVE POSITIVE MATCHES?
# ============================================================

positive_by_entity = (
    df.groupby("source1_entity_id")["label"]
    .sum()
)

positive_entities = (
    positive_by_entity[
        positive_by_entity > 0
    ]
)

print(
    "S1 entities with >=1 positive:",
    len(positive_entities)
)

print(
    "S1 entities with 0 positives:",
    len(entities) - len(positive_entities)
)


# ============================================================
# 5. POSITIVE MATCH DISTRIBUTION
# ============================================================

print("\nPositive matches per S1 entity:")

print(
    positive_entities
    .value_counts()
    .sort_index()
)


# ============================================================
# 6. ENTITY-LEVEL TRAIN / VALIDATION SPLIT
# ============================================================

train_entities, val_entities = train_test_split(
    entities,
    test_size=0.2,
    random_state=42
)


train_df = df[
    df["source1_entity_id"].isin(
        train_entities
    )
]

val_df = df[
    df["source1_entity_id"].isin(
        val_entities
    )
]


# ============================================================
# 7. REPORT
# ============================================================

print("\n")
print("=" * 60)
print("ENTITY-AWARE SPLIT")
print("=" * 60)

print(
    "\nTraining S1 entities:",
    len(train_entities)
)

print(
    "Validation S1 entities:",
    len(val_entities)
)

print(
    "\nTraining candidate pairs:",
    len(train_df)
)

print(
    "Validation candidate pairs:",
    len(val_df)
)

print(
    "\nTraining positive pairs:",
    train_df["label"].sum()
)

print(
    "Validation positive pairs:",
    val_df["label"].sum()
)

print(
    "\nTraining positive entities:",
    (
        train_df.groupby("source1_entity_id")["label"]
        .sum() > 0
    ).sum()
)

print(
    "Validation positive entities:",
    (
        val_df.groupby("source1_entity_id")["label"]
        .sum() > 0
    ).sum()
)


# ============================================================
# 8. CHECK FOR LEAKAGE
# ============================================================

overlap = (
    set(train_entities)
    &
    set(val_entities)
)

print(
    "\nEntity overlap between train and validation:",
    len(overlap)
)

if len(overlap) == 0:
    print("✓ No entity leakage")
else:
    print("WARNING: Entity leakage detected")


print("\nDone.")