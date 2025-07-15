# name-cleaning-mapping-main

This project provides a robust pipeline for cleaning, standardizing, de-duplicating, and uniquely identifying names (and titles) from Canadian government datasets. The workflow ensures consistent, canonical naming for individuals, tracks all variants, leverages metadata to avoid false merges, and assigns a stable UUID to each unique individual for long-term traceability.

## Workflow Summary

Initial Cleaning: Normalize raw names, remove honorifics/titles, fix encoding, and standardize format.
Phase 1 Clustering: Group similar name variants using fuzzy matching, merge rare forms, and pick canonical names/titles.
Phase 2 Superclustering: Merge clusters further using name similarity plus metadata (title, date, initials, etc.), ensuring only compatible clusters merge.
Reference Dictionary: Apply manual corrections using a curated mapping of "wrong → correct" names.
UUID Assignment: Assign and persist a stable UUID to each unique, canonical name for identity tracking.

project-root/
├── input/
│   ├── dpoh_names_title_all.csv            # Raw input data
│   ├── dpoh_names_title_A.csv, ...         # Split by first letter after initial clean
├── output/
│   ├── Clean/                              # After phase 1 cleaning, CSVs are stored here
│   ├── JSON/                               # After phase 1 cleaning, JSONs are stored here
│   ├── Clean/                              # After phase 1 cleaning, Summary files are stored here
│   ├── Final-CSV/                          # After phase 2 superclustering, , CSVs are stored here
│   ├── Final-JSON/                         # Cluster info as JSONs (phase 2)
│   ├── Final-Summary/                      # Summary tables per cluster(phase 2)
│   ├── UUID-CSV/                           # CSVs with UUID columns
│   ├── UUID-JSON/                          # JSONs with UUID in unique_id field
│   ├── dpoh_names_title_all_cleaned.csv/   # After initail cleaning, this master file is sliced into 26 files for phases 1 and 2
│   └── dpoh_mapping.json                   # Canonical name → UUID mapping
├── Version1_Cleaning.py                    # Initial cleaning script
├── name_cleaner.py                         # Phase 1 cleaning/clustering
├── name_cleanerv1.py                       # Script to run Phase 1 cleaning/clustering
├── name_cleanerv2.py                       # Phase 2 superclustering
├── dpoh_id_creator.py                      # UUID generation/mapping
└── Reference Dictionary.xlsx               # Manual corrections dictionary   


## Detailed Steps

1. Initial Cleaning
Goal: Remove non-essential words/titles, standardize formatting, and split files by first letter for efficiency. It also gives a master file without the slicing.
Logic:
- Fix encoding and strip accents.
- Remove prefixes/suffixes (e.g., “The Honourable”, “Minister”, “PhD”, “Prime Minister’s Office”).
- Remove punctuation, extra whitespace, content in brackets, and other noise.
- Capitalize correctly.
Input: input/dpoh_names_title_all.csv 
Output: CleanName_version1, CleanTitle_version1 columns. input/dpoh_names_title_A.csv, ... , output/dpoh_names_title_all_cleaned.csv

2. Phase 1: Canonical Name Clustering
Goal: Group all spelling variants of the same person together and pick the most common form.
Logic:
- Block clustering: Use the first letter and soundex for speed.
- Levenshtein distance: Names within a threshold (based on length) are grouped.
- Rare variant merging: One-off spellings are merged more aggressively.
- Aggressive merge for small clusters: Very small clusters are double-checked for potential merges.
- Canonical name/title selection: Most frequent variant is chosen as the final.
- Reference dictionary applied: Corrects frequent known mistakes.
Input: input/dpoh_names_title_A.csv...
Outputs: CleanName_version2, CleanTitle_version2 columns in CSV.
JSON summary of clusters, including all variants, counts, and a placeholder for UUID.
output/Clean/cleaned_A.csv,...; output/JSON/cleaned_A_uniquenames.json,...; output/Summary/cleaned_A_summary.csv,...

3. Phase 2: Superclustering with Metadata
Goal: Catch any clusters missed in Phase 1 by leveraging metadata.
Logic:
- Group by CleanName_version2 and aggregate all linked raw names/titles.
- Use Levenshtein and metadata (title, date ranges, initials) to merge compatible clusters.
- Pick the most frequent final form as CleanName_version3, and the most common title as CleanTitle_version3.
- Output summary tables for each cluster, including all raw name variants.
- Reference Dictionary: Manual corrections are again applied here.
Input: output/Clean/cleaned_A.csv,...
Output: CleanName_version3, CleanTitle_version3 columns in CSV.
output/Final-CSV/Supercleaned_A.csv,...; output/Final-JSON/Supercleaned_A_uniquenames.json,...; output/Final-Summary/Supercleaned_A_summary.csv,...


4. UUID Assignment & Final Outputs
Goal: Assign a stable, unique 16-character identifier to each unique name for persistent identity tracking.
Logic:
- A global mapping file ensures the same name always gets the same UUID across all files/runs.
- Each row in the CSVs gets a dpoh_id column.
- In JSON cluster files, the UUID is written to the "unique_id" field (overwriting the placeholder).
Input: output/Final-CSV/Supercleaned_A.csv,...; output/Final-JSON/Supercleaned_A_uniquenames.json,...
Outputs: All cleaned CSVs and JSONs now contain UUIDs.
output/UUID-JSON/Supercleaned_A_uniquenames.json,...; output/UUID-CSV/Supercleaned_A_uuid.csv,...
output/dpoh_mapping.json for cross-file consistency.

Reference Dictionary: Reference Dictionary.xlsx is a curated list of known misspellings or alternate forms, mapping each “Wrong Name” to the “Correct Name.” Applied at multiple phases to catch stubborn or edge-case errors.


## Key Algorithms & Logic
- String normalization: Lowercasing, accent removal, token filtering.
- Prefix/suffix stripping: Extensive lists, iteratively expanded.
- Fuzzy matching: Levenshtein distance, phonetic similarity (soundex).
- Cluster merging: Both frequency and metadata-based, with safeguards against unrelated merges.
- Canonical selection: Always “the most common full name wins” in a cluster.
- UUID assignment: Persistent, repeatable, and global across the project.

## How to Run
1. Initial Clean: python Version1_Cleaning.py
   This reads the input/dpoh_names_title_all.csv and saves the output to: input/dpoh_names_title_A.csv, ... , output/dpoh_names_title_all_cleaned.csv 
2. Phase 1 Cleaning:
   - To run the script for all all 26 files: python name_cleanerv1.py
     This reads the input/dpoh_names_title_A.csv, ... and saves the output to: output/Clean/cleaned_A.csv,...; output/JSON/cleaned_A_uniquenames.json,...; output/Summary/cleaned_A_summary.csv,...
   - To run a particualr file(Example: A): python name_cleaner.py   -i input/dpoh_names_title_A.csv.csv   -o output/cleaned_A.csv
     This reads the input/dpoh_names_title_A.csv and saves the output to: output/cleaned_A.csv,...; output/cleaned_A_uniquenames.json,...; output/cleaned_A_summary.csv,...
     These 3 files will have to be dragged and dropped in the respective folders for further cleaning
   - (IMPORTANT) To run the entire all cleaned file: python name_cleaner.py   -i output/dpoh_names_title_all_cleaned.csv   -o output/cleaned_all.csv
     This reads the output/dpoh_names_title_all_cleaned.csv and saves the output to output/cleaned_all.csv,...; output/cleaned_all_uniquenames.json,...; output/cleaned_all_summary.csv,...
     These 3 files will have to be dragged and dropped in the respective folders for further cleaning
3. Phase 2 Cleaning:
  - To run the script for all all 26 files: python name_cleanerv2.py
     This reads the output/Clean/cleaned_A.csv, ... and saves the output to: output/Final-CSV/Supercleaned_A.csv,...; output/Final-JSON/Supercleaned_A_uniquenames.json,...; output/Final-Summary/Supercleaned_A_summary.csv,...
  - To run a particualr file(Example: A): python name_cleanerv2.py -l A
     This reads the output/Clean/cleaned_A.csv and saves the output to: output/Final-CSV/Supercleaned_A.csv; output/Final-JSON/Supercleaned_A_uniquenames.json; output/Final-Summary/Supercleaned_A_summary.csv
  - (IMPORTANT) To run the entire all cleaned file: 
4. UUID Generation: python dpoh_id_creator.py
  This reads the output/Final-CSV/Supercleaned_A.csv,...; output/Final-JSON/Supercleaned_A_uniquenames.json,...; and saves the output to: output/UUID-JSON/Supercleaned_A_uniquenames.json,...; output/UUID-CSV/Supercleaned_A_uuid.csv,...; output/dpoh_mapping.json
