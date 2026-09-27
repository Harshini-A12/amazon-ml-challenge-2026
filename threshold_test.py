import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading feature dataset...")

df = pd.read_csv(
    "prototype_features_v2.csv"
)

feature_columns = [
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

X = df[feature_columns]
y = df["label"]


# ============================================================
# 2. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. TRAIN MODEL
# ============================================================

print("Training model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 4. PREDICT PROBABILITIES
# ============================================================

probabilities = model.predict_proba(
    X_val
)[:, 1]


# ============================================================
# 5. TEST DIFFERENT THRESHOLDS
# ============================================================

thresholds = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99,
    0.995,
    0.999
]


print("\n")
print("=" * 75)
print("THRESHOLD COMPARISON")
print("=" * 75)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'F0.5':<15}"
    f"{'Predicted +':<15}"
)

print("-" * 75)


for threshold in thresholds:

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

    f05 = fbeta_score(
        y_val,
        predictions,
        beta=0.5,
        zero_division=0
    )

    predicted_positive = predictions.sum()

    print(
        f"{threshold:<12.3f}"
        f"{precision:<15.4f}"
        f"{recall:<15.4f}"
        f"{f05:<15.4f}"
        f"{predicted_positive:<15}"
    )


print("\nDone.")