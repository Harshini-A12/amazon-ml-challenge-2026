import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================
# 1. LOAD FEATURES
# ============================================================

print("Loading feature dataset...")

df = pd.read_csv(
    "prototype_features_v2.csv"
)

print("Rows:", len(df))


# ============================================================
# 2. FEATURES
# ============================================================

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


print("\nClass distribution:")
print(y.value_counts())


# ============================================================
# 3. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_val))


# ============================================================
# 4. MODEL
# ============================================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 5. PREDICTION
# ============================================================

probabilities = model.predict_proba(
    X_val
)[:, 1]


# Default threshold
predictions = (
    probabilities >= 0.5
).astype(int)


# ============================================================
# 6. METRICS
# ============================================================

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


print("\n========================================")
print("BASELINE MODEL RESULTS")
print("========================================")

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1 Score:  {f1:.4f}"
)


# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================

print("\n========================================")
print("MODEL COEFFICIENTS")
print("========================================")

for feature, coefficient in zip(
    feature_columns,
    model.coef_[0]
):

    print(
        f"{feature:25s} "
        f"{coefficient: .4f}"
    )


print("\nDone.")