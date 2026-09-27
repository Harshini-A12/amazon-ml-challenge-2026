import pandas as pd


def analyze_file(path, label):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    df = pd.read_csv(
        path,
        sep="\t",
        nrows=10000
    )

    print("Rows sampled:", len(df))

    print("\nMissing values:")
    print(df[["business_name", "business_address"]].isna().sum())

    print("\nAverage character length:")

    print(
        "Business name:",
        df["business_name"].astype(str).str.len().mean()
    )

    print(
        "Business address:",
        df["business_address"].fillna("").astype(str).str.len().mean()
    )

    print("\nExamples:")

    for _, row in df.head(5).iterrows():
        print("Name   :", row["business_name"])
        print("Address:", row["business_address"])
        print()


analyze_file(
    "Dataset/Train/train_source1.tsv",
    "TRAIN SOURCE 1"
)

analyze_file(
    "Dataset/Train/train_source2.tsv",
    "TRAIN SOURCE 2"
)

analyze_file(
    "Dataset/Train/train_source3.tsv",
    "TRAIN SOURCE 3"
)