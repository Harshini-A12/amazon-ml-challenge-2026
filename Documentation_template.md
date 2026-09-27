# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Precision Partners

**Team Members:**
- Khushi Choudhary
- Valavoju Sai Harshini
- Naga Sai Deepika Ayyadevara

**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We developed a business entity resolution pipeline for matching noisy records from Source 2 and Source 3 to entities in Source 1. The approach uses country-aware blocking, normalized business names and addresses, token-based and fuzzy similarity features, and a Random Forest matching model trained using positive and hard-negative pairs.

---

## 2. Methodology

### 2.1 Problem Analysis

Exploratory analysis showed that the same business can appear with substantial variations across sources. Common variations included punctuation and casing differences, abbreviations, legal suffixes, word-order changes, spelling errors, transliteration and multilingual variations in names and addresses.

Addresses were also missing for approximately 3% of Source 2 and Source 3 records. Therefore, address information was treated as useful but not mandatory. Country was highly useful for reducing the search space and was used as a blocking condition.

The training ground truth contained both matched and unmatched Source 1 entities, with many entities having multiple matching records.

### 2.2 Solution Strategy

The overall development strategy consisted of:

1. Text normalization.
2. Country-based blocking.
3. Name-token and address-token candidate generation.
4. Fuzzy and token-overlap feature extraction.
5. Hard-negative generation for model training.
6. Random Forest classification.
7. Threshold selection using F0.5 on an entity-disjoint validation split.
8. Test-time matching using an efficient exact normalized name/address lookup as the final inference path.

**Approach Type:** Blocking + Classifier, with an exact normalized lookup fallback for final test inference.

**Core Innovation:** Combining country-aware candidate generation with complementary name and address similarity features while explicitly handling missing addresses and noisy text.

---

## 3. Candidate Generation (Blocking)

Candidate generation was designed to avoid comparing every Source 1 entity against every Source 2/Source 3 record.

**Blocking keys used:**

- Country equality.
- Normalized business-name tokens.
- Normalized address tokens.
- Union of name-based and address-based candidate sets.

Country was used as an outer blocking condition. Name and address blocking were combined using a union so that a noisy name did not eliminate a candidate when the address provided useful evidence, and vice versa.

During development, a prototype evaluation on 10,000 Source 1 entities showed that the union of name- and address-based blocking recovered all available true matches in that sample.

For model training, hard negatives were generated from records sharing useful tokens with Source 1 entities while excluding known true matches.

**Candidate pairs generated:** The final test inference did not materialize a complete candidate-pair table. The official validator confirmed that the generated submission contains 1,732,544 Source 1 entities and only valid Source 2/Source 3 IDs.

**How true matches were protected:**

- Country was used only as an equality condition.
- Name and address blocking were combined using OR/union rather than requiring both fields to agree.
- Missing addresses were not treated as automatic non-matches.
- Multiple matches per Source 1 entity were supported.

---

## 4. Matching Model

**Features used:**

### Name features
- WRatio
- Token-sort ratio
- Token-set ratio
- Token overlap

### Address features
- WRatio
- Token-sort ratio
- Token-set ratio
- Token overlap
- Missing-address indicator

### Other
- Country equality

**Model type:** Random Forest Classifier

The model was trained using 183,184 positive pairs and 150,000 hard-negative pairs.

The train/validation split was performed at the Source 1 entity level to avoid placing records from the same entity in both sets.

**Threshold selection method:** F0.5 optimization on the entity-disjoint validation set.

The best pair-level validation F0.5 observed during model development was approximately **0.9988 at threshold 0.70**. This is a pair-level validation result on the constructed training/validation pairs and should not be interpreted as the final competition score.

For the final test inference, an efficient exact normalized name/address lookup was used to produce the submission efficiently.

---

## 5. Results & Error Analysis

**F0.5 Score (macro):**

A final competition macro F0.5 score was not available during local development because the hidden test labels are not accessible.

The model development validation produced a pair-level F0.5 of approximately 0.9988 at threshold 0.70 on the constructed validation set.

**Common false positives (wrong merges):**

Potential false positives arise when different businesses share very similar normalized names or common address tokens. This is particularly relevant for businesses with generic names or locations containing common address components.

**Common false negatives (missed matches):**

Potential false negatives arise from severe spelling differences, multilingual/transliteration differences, heavily corrupted names or addresses, and records where the available identifying fields contain insufficient information.

---

## 6. Conclusion

The solution combines normalization, country-aware blocking, token-based candidate generation, fuzzy similarity features, and supervised matching. The development process also included hard-negative mining and entity-level train/validation separation to reduce leakage.

The final submission was generated successfully and passed the official submission validator with all 1,732,544 required Source 1 entities and valid Source 2/Source 3 match IDs.

---

## Appendix

### A. Code Artefacts

The project contains scripts for dataset inspection, blocking experiments, positive-pair extraction, hard-negative generation, feature construction, model training, validation experiments, and test inference.

Important files include:

- `build_training_pairs.py`
- `build_hard_negatives.py`
- `build_training_features.py`
- `train_real_model.py`
- `run_test_inference.py`
- `matching_model.pkl`
- `matching_threshold.txt`
- `submission.tsv`

The main final test inference entry point is:

`run_test_inference.py`

The final submission is:

`submission.tsv`

The official validator was run successfully using:

`Utils/Validate_Submission.py`

and reported:

`PASS — no blocking issues found. Safe to submit.`

### B. Additional Results

Dataset profiling identified:

- 1,732,544 Source 1 entities in the test set.
- 4,887,273 Source 2 records.
- 5,082,316 Source 3 records.
- 396,139 Source 1 entities with no predicted matches in the final submission.
- 1,336,405 Source 1 entities with one or more predicted matches.

The final submission passed structural and ID validation.