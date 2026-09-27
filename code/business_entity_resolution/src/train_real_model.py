import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    classification_report
)


# ============================================================
# 1. LOAD FEATURES
# ============================================================

print("Loading training features...")

df = pd.read_csv("training_features.csv")

print("Total rows:", len(df))


# ============================================================
# 2. ENTITY-AWARE SPLIT
# ============================================================

print("\nCreating entity-aware split...")

entities = df["source1_entity_id"].unique().tolist()

train_entities, val_entities = train_test_split(
    entities,
    test_size=0.20,
    random_state=42
)

train_df = df[
    df["source1_entity_id"].isin(train_entities)
].copy()

val_df = df[
    df["source1_entity_id"].isin(val_entities)
].copy()

print("Training entities:", len(train_entities))
print("Validation entities:", len(val_entities))

print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))

print(
    "Entity overlap:",
    len(
        set(train_entities) &
        set(val_entities)
    )
)


# ============================================================
# 3. FEATURES
# ============================================================

FEATURES = [
    "name_wratio",
    "name_token_sort",
    "name_token_set",
    "name_token_overlap",
    "address_wratio",
    "address_token_sort",
    "address_token_set",
    "address_token_overlap",
    "address_missing",
    "country_equal"
]

X_train = train_df[FEATURES]
y_train = train_df["label"]

X_val = val_df[FEATURES]
y_val = val_df["label"]


# ============================================================
# 4. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# 5. PREDICT PROBABILITIES
# ============================================================

print("\nGenerating validation probabilities...")

probabilities = model.predict_proba(
    X_val
)[:, 1]


# ============================================================
# 6. THRESHOLD TESTING
# ============================================================

print("\n")
print("=" * 70)
print("THRESHOLD EVALUATION")
print("=" * 70)

best_f05 = -1
best_threshold = None

for threshold in [
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99
]:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    f05 = fbeta_score(
        y_val,
        predictions,
        beta=0.5,
        zero_division=0
    )

    predicted_positive = predictions.sum()

    print(
        f"Threshold {threshold:.2f} | "
        f"Precision {precision:.4f} | "
        f"Recall {recall:.4f} | "
        f"F1 {f1:.4f} | "
        f"F0.5 {f05:.4f} | "
        f"Predicted + {predicted_positive}"
    )

    if f05 > best_f05:
        best_f05 = f05
        best_threshold = threshold


# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================

print("\n")
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(importance.to_string(index=False))


# ============================================================
# 8. BEST THRESHOLD SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print("Best threshold:", best_threshold)
print("Best pair-level F0.5:", best_f05)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving model...")

joblib.dump(
    model,
    "matching_model.pkl"
)

with open("matching_threshold.txt", "w") as f:
    f.write(str(best_threshold))

print("Model saved: matching_model.pkl")
print("Threshold saved:", best_threshold)

print("\nDone.")