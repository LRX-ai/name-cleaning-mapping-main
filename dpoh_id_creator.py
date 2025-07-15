import os
import glob
import pandas as pd
import uuid
import json

# --- Configuration ---
base_dir = 'output'
input_csv_dir = os.path.join(base_dir, 'Final-CSV')
input_json_dir = os.path.join(base_dir, 'Final-JSON')
uuid_csv_dir = os.path.join(base_dir, 'UUID-CSV')
uuid_json_dir = os.path.join(base_dir, 'UUID-JSON')
mapping_file = os.path.join(base_dir, 'dpoh_mapping.json')

for d in [uuid_csv_dir, uuid_json_dir]:
    os.makedirs(d, exist_ok=True)

# --- 1. Load existing mapping (or start empty) ---
if os.path.exists(mapping_file):
    with open(mapping_file, 'r', encoding='utf-8') as f:
        dpoh_mapping = json.load(f)
else:
    dpoh_mapping = {}

# --- 2. Process all Supercleaned CSVs ---
csv_files = glob.glob(os.path.join(input_csv_dir, 'Supercleaned_*.csv'))
all_dfs = []
for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    if 'CleanName_version3' not in df.columns:
        print(f"[WARNING] File missing CleanName_version3 column: {csv_path}")
        continue
    all_dfs.append(df)

if not all_dfs:
    print("[ERROR] No input CSVs found or none with CleanName_version3. Exiting.")
    exit(1)

combined = pd.concat(all_dfs, ignore_index=True)

# --- 3. Generate/reuse UUID for each CleanName_version3 ---
all_names = pd.Series(combined['CleanName_version3'].dropna().unique())
for name in all_names:
    if name not in dpoh_mapping:
        dpoh_mapping[name] = uuid.uuid4().hex[:16]

# --- 4. Write back mapping file ---
with open(mapping_file, 'w', encoding='utf-8') as f:
    json.dump(dpoh_mapping, f, indent=2, ensure_ascii=False)

# --- 5. Process each file: Add UUID to CSV and JSON ---
for csv_path in csv_files:
    basename = os.path.basename(csv_path)
    letter = basename.split('_')[-1].split('.')[0]  # e.g. Supercleaned_A.csv → A

    # CSV
    df = pd.read_csv(csv_path)
    if 'CleanName_version3' not in df.columns:
        print(f"[WARNING] File missing CleanName_version3 column: {csv_path}")
        continue
    df['dpoh_id'] = df['CleanName_version3'].map(dpoh_mapping)
    # Save to UUID-CSV
    out_csv = os.path.join(uuid_csv_dir, basename)
    df.to_csv(out_csv, index=False)
    print(f"[CSV] Wrote {out_csv}")

    # JSON
    json_path = os.path.join(input_json_dir, f"Supercleaned_{letter}_uniquenames.json")
    uuid_json_path = os.path.join(uuid_json_dir, f"Supercleaned_{letter}_uniquenames.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            clusters = json.load(f)
        for cluster in clusters:
            # Set UUID as unique_id
            cname = cluster.get('cleaned_name', '')
            cluster['unique_id'] = dpoh_mapping.get(cname, '')
        with open(uuid_json_path, 'w', encoding='utf-8') as f:
            json.dump(clusters, f, indent=2, ensure_ascii=False)
        print(f"[JSON] Wrote {uuid_json_path}")
    else:
        print(f"[WARNING] JSON not found for {letter}: {json_path}")

print(f"All UUID-enhanced CSV and JSON files written to {uuid_csv_dir} and {uuid_json_dir}")
print(f"Global mapping written to {mapping_file}")
