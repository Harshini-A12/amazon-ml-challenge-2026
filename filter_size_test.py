import pandas as pd
import re
import unicodedata


# ============================================================
# SETTINGS
# ============================================================

BASE = "Dataset"

S1_PATH = f"{BASE}/train/train_source1.tsv"
S2_PATH = f"{BASE}/train/train_source2.tsv"
S3_PATH = f"{BASE}/train/train_source3.tsv"
GT_PATH = f"{BASE}/train/train_ground_truth.tsv"

N = 10000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = "".join(
        " " if unicodedata.category(char).startswith("P")
        else char
        for char in text
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


STOPWORDS = {
    "inc", "incorporated",
    "llc",
    "ltd", "limited",
    "corp", "corporation",
    "company", "co",
    "private", "pvt",
    "plc"
}


def name_tokens(text):

    return {
        token
        for token in normalize_text(text).split()
        if len(token) > 1
        and token not in STOPWORDS
    }


def address_tokens(text):

    return {
        token
        for token in normalize_text(text).split()
        if len(token) > 1
    }


# ============================================================
# LOAD
# ============================================================

print("Loading data...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    nrows=N
)

s2 = pd.read_csv(
    S2_PATH,
    sep="\t",
    nrows=N
)

s3 = pd.read_csv(
    S3_PATH,
    sep="\t",
    nrows=N
)

gt = pd.read_csv(
    GT_PATH,
    sep="\t"
)

sources = pd.concat(
    [s2, s3],
    ignore_index=True
)

print("S1:", len(s1))
print("S2 + S3:", len(sources))


# ============================================================
# PREPARE SOURCE RECORDS
# ============================================================

print("\nPreparing source records...")

source_records = {}

for _, row in sources.iterrows():

    entity_id = row["entity_id"]

    source_records[entity_id] = {
        "country": row["country"],
        "name_tokens": name_tokens(
            row["business_name"]
        ),
        "address_tokens": address_tokens(
            row["business_address"]
        )
    }


# ============================================================
# BUILD INDEXES
# ============================================================

print("Building indexes...")

name_index = {}
address_index = {}

for entity_id, record in source_records.items():

    country = record["country"]

    for token in record["name_tokens"]:

        name_index.setdefault(
            (country, token),
            set()
        ).add(entity_id)

    for token in record["address_tokens"]:

        address_index.setdefault(
            (country, token),
            set()
        ).add(entity_id)


# ============================================================
# GROUND TRUTH
# ============================================================

print("Preparing ground truth...")

ground_truth = {}

for _, row in gt.iterrows():

    value = str(
        row["matched_entity_ids"]
    )

    if value == "nan" or value == "":
        ground_truth[
            row["source1_entity_id"]
        ] = set()
    else:
        ground_truth[
            row["source1_entity_id"]
        ] = {
            x.strip()
            for x in value.split(",")
            if x.strip()
        }


# ============================================================
# FILTER TEST
# ============================================================

filters = {
    "baseline >=1 OR >=1": lambda n, a:
        n >= 1 or a >= 1,

    "name >=2 OR address >=2": lambda n, a:
        n >= 2 or a >= 2,

    "name >=1 OR address >=2": lambda n, a:
        n >= 1 or a >= 2,

    "name >=2 OR address >=1": lambda n, a:
        n >= 2 or a >= 1,

    "name >=3 OR address >=2": lambda n, a:
        n >= 3 or a >= 2,

    "name >=2 OR address >=3": lambda n, a:
        n >= 2 or a >= 3,
}


# Statistics

total_candidates = 0

kept = {
    name: 0
    for name in filters
}

true_total = 0

true_kept = {
    name: 0
    for name in filters
}


# ============================================================
# PROCESS
# ============================================================

print("\nTesting filters...\n")

for count, (_, row) in enumerate(
    s1.iterrows(),
    start=1
):

    country = row["country"]

    s1_name = name_tokens(
        row["business_name"]
    )

    s1_address = address_tokens(
        row["business_address"]
    )


    # Original blocking candidates

    candidates = set()

    for token in s1_name:

        candidates.update(
            name_index.get(
                (country, token),
                set()
            )
        )

    for token in s1_address:

        candidates.update(
            address_index.get(
                (country, token),
                set()
            )
        )


    total_candidates += len(candidates)


    # Ground truth

    true_matches = ground_truth.get(
        row["entity_id"],
        set()
    )

    true_total += len(true_matches)


    # Test every candidate

    for candidate_id in candidates:

        candidate = source_records[
            candidate_id
        ]

        shared_name = len(
            s1_name &
            candidate["name_tokens"]
        )

        shared_address = len(
            s1_address &
            candidate["address_tokens"]
        )


        for filter_name, rule in filters.items():

            if rule(
                shared_name,
                shared_address
            ):

                kept[filter_name] += 1

                if candidate_id in true_matches:

                    true_kept[
                        filter_name
                    ] += 1


    if count % 1000 == 0:

        print(
            f"Processed {count}/{N}"
        )


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FILTER COMPARISON")
print("=" * 70)

print(
    f"\nBaseline candidate pairs: "
    f"{total_candidates:,}"
)

print(
    f"Average candidates per S1: "
    f"{total_candidates / N:.2f}"
)

print(
    f"\nTrue-match pairs found in prototype: "
    f"{true_total:,}"
)


for filter_name in filters:

    candidate_count = kept[
        filter_name
    ]

    recall = (
        true_kept[filter_name]
        / true_total
        * 100
        if true_total > 0
        else 0
    )

    reduction = (
        1 -
        candidate_count /
        total_candidates
    ) * 100

    print("\n" + "-" * 70)

    print(filter_name)

    print(
        f"Candidates: "
        f"{candidate_count:,}"
    )

    print(
        f"Average per S1: "
        f"{candidate_count / N:.2f}"
    )

    print(
        f"Candidate reduction: "
        f"{reduction:.2f}%"
    )

    print(
        f"True-match recall: "
        f"{true_kept[filter_name]:,}"
        f"/{true_total:,}"
        f" = {recall:.2f}%"
    )


print("\nDone.")