# Amazon ML Challenge 2026 — Business Entity Resolution

## Overview

This project solves the business entity resolution problem by matching records from Source 2 and Source 3 to Source 1.

The solution uses country-aware matching and normalized business names and addresses. A Random Forest model was also trained using name/address similarity features and hard-negative examples.

For the final test inference, an efficient exact normalized name/address matching approach was used because of the large dataset size.

## Dataset

The expected dataset structure is:

dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv

All files are tab-separated TSV files.

## Source Code

src/
├── train_real_model.py
└── run_test_inference.py

### train_real_model.py

Trains the Random Forest matching model using the prepared training feature dataset.

### run_test_inference.py

Processes the test Source 1, Source 2, and Source 3 data using chunked processing and country-aware normalized name/address matching.

## Final Output

The final submission contains:

output/
├── matching_results.tsv
└── candidate_pairs.tsv

`matching_results.tsv` contains one row for every Source 1 entity.

Entities with no identified matches have an empty match list.

## Dependencies

Python 3.10+ is recommended.

Main dependencies:

- pandas
- scikit-learn
- rapidfuzz
- joblib

## Important

No external data lookup or external entity information is used.

The final inference uses chunked processing to handle the large dataset while controlling memory usage.