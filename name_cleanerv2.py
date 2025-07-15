import pandas as pd
import numpy as np
from collections import Counter
from Levenshtein import distance as lev
import json
from pathlib import Path

ref_dict_path = "Reference Dictionary.xlsx"

# Directories
input_dir = Path("output/Clean")
csv_dir = Path("output/Final-CSV")
json_dir = Path("output/Final-JSON")
summary_dir = Path("output/Final-Summary")
for d in [csv_dir, json_dir, summary_dir]:
    d.mkdir(parents=True, exist_ok=True)

def to_py_type(obj):
    if isinstance(obj, dict):
        return {k: to_py_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_py_type(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def initials(name):
    parts = name.split()
    first = parts[0][0] if parts and len(parts[0]) > 0 else ""
    last = parts[-1][0] if len(parts) > 1 and len(parts[-1]) > 0 else ""
    return (first, last)

def similar_titles(title1, title2):
    if not title1 or not title2:
        return True
    if title1 == title2:
        return True
    if title1 in title2 or title2 in title1:
        return True
    if lev(title1, title2) <= 3:
        return True
    return False

def date_overlap(min1, max1, min2, max2):
    # Handles overlap or within 3 years (1095 days)
    try:
        if pd.notnull(min1) and pd.notnull(max1) and pd.notnull(min2) and pd.notnull(max2):
            if max(min1, min2) <= min(max1, max2):
                return True
            if abs((min1 - max2).days) <= 1095 or abs((min2 - max1).days) <= 1095:
                return True
    except Exception:
        return True
    return False

def superclean_letter(letter, ref_map):
    input_csv = input_dir / f"cleaned_{letter}.csv"
    if not input_csv.exists():
        print(f"Input file missing for {letter}, skipping.")
        return

    output_csv = csv_dir / f"Supercleaned_{letter}.csv"
    output_json = json_dir / f"Supercleaned_{letter}_uniquenames.json"
    summary_csv = summary_dir / f"Supercleaned_{letter}_summary.csv"

    df = pd.read_csv(input_csv)
    if 'dates' in df.columns:
        df['dates'] = pd.to_datetime(df['dates'], errors='coerce')

    # --- Aggregate by CleanName_version2 ---
    agg = df.groupby("CleanName_version2").agg({
        "CleanName_version1": lambda x: list(x),
        "dpoh_name_raw": lambda x: list(x),
        "CleanTitle_version2": lambda x: list(x),
        "dates": ["min", "max", "count"]
    })
    agg.columns = ['variants', 'raw_names', 'titles', 'first_appearance', 'last_appearance', 'count']
    agg = agg.reset_index()

    # --- Metadata-aware clustering ---
    parent = list(range(len(agg)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i, row1 in agg.iterrows():
        for j, row2 in agg.iterrows():
            if i >= j:
                continue
            name1 = str(row1['CleanName_version2']).strip().lower()
            name2 = str(row2['CleanName_version2']).strip().lower()
            # 1. Levenshtein similarity (2 or less)
            if lev(name1, name2) > 2:
                continue
            # 2. Initials must match
            if initials(name1) != initials(name2):
                continue
            # 3. Title similarity (most frequent)
            title1 = Counter(row1['titles']).most_common(1)[0][0].lower() if row1['titles'] else ""
            title2 = Counter(row2['titles']).most_common(1)[0][0].lower() if row2['titles'] else ""
            if not similar_titles(title1, title2):
                continue
            # 4. Date overlap
            min1, max1 = row1['first_appearance'], row1['last_appearance']
            min2, max2 = row2['first_appearance'], row2['last_appearance']
            if not date_overlap(min1, max1, min2, max2):
                continue
            # If ALL checks passed, union the clusters
            union(i, j)

    # --- Assign new canonical names/titles for clusters ---
    clusters = {}
    for idx in range(len(agg)):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    name2_to_v3 = {}
    name2_to_title3 = {}
    cluster_jsons = []
    summary_rows = []
    for inds in clusters.values():
        all_names = sum([agg.loc[i, 'variants'] for i in inds], [])
        all_titles = sum([agg.loc[i, 'titles'] for i in inds], [])
        all_raw_names = sum([agg.loc[i, 'raw_names'] for i in inds], [])
        all_first = [agg.loc[i, 'first_appearance'] for i in inds if pd.notnull(agg.loc[i, 'first_appearance'])]
        all_last = [agg.loc[i, 'last_appearance'] for i in inds if pd.notnull(agg.loc[i, 'last_appearance'])]
        all_count = sum([agg.loc[i, 'count'] for i in inds])
        v3_name = Counter(all_names).most_common(1)[0][0]
        v3_title = Counter(all_titles).most_common(1)[0][0] if all_titles else ""
        all_raw_names_str = ", ".join(sorted(set(all_raw_names)))
        for i in inds:
            name2_to_v3[agg.loc[i, 'CleanName_version2']] = v3_name
            name2_to_title3[agg.loc[i, 'CleanName_version2']] = v3_title

        # --- JSON summary for this cluster ---
        cluster_jsons.append({
            "unique_id": "",
            "cleaned_name": v3_name,
            "first_appearance": str(min(all_first).date()) if all_first else "",
            "last_appearance": str(max(all_last).date()) if all_last else "",
            "title": v3_title,
            "Count": all_count,
            "variants": [
                {"raw_name": n, "count": all_names.count(n)}
                for n in sorted(set(all_names))
            ],
            "dpoh_name_raw": all_raw_names_str
        })

        # --- CSV summary row for this cluster ---
        summary_rows.append({
            "CleanName_version3": v3_name,
            "CleanTitle_version3": v3_title,
            "first_appearance": min(all_first).date() if all_first else "",
            "last_appearance": max(all_last).date() if all_last else "",
            "count": all_count,
            "dpoh_name_raw": all_raw_names_str
        })

    # --- Add CleanName_version3/CleanTitle_version3 columns to original DataFrame ---
    df['CleanName_version3'] = df['CleanName_version2'].map(name2_to_v3)
    df['CleanTitle_version3'] = df['CleanName_version2'].map(name2_to_title3)

    # --- Apply Reference Dictionary for CleanName_version3 ---
    if ref_map:
        df['CleanName_version3'] = df['CleanName_version3'].astype(str).str.strip().replace(ref_map)
        for entry in cluster_jsons:
            n3 = entry['cleaned_name']
            if n3 in ref_map:
                entry['cleaned_name'] = ref_map[n3]
        for row in summary_rows:
            n3 = row['CleanName_version3']
            if n3 in ref_map:
                row['CleanName_version3'] = ref_map[n3]
        print(f"Applied reference dictionary corrections to CleanName_version3 for {letter}.")

    # --- Save to main CSV ---
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")

    # --- Save summary CSV ---
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")

    # --- Numpy to python types for JSON ---
    cluster_jsons_py = to_py_type(cluster_jsons)

    # --- Save to JSON ---
    with open(output_json, "w") as f:
        json.dump(cluster_jsons_py, f, indent=2)
    print(f"Wrote {output_json}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--letter', type=str, help='Specify a single letter (A-Z) to process. If omitted, runs all.')
    args = parser.parse_args()

    # Load reference dictionary once
    if Path(ref_dict_path).exists():
        ref_df = pd.read_excel(ref_dict_path)
        ref_map = dict(zip(ref_df['Wrong Name'].astype(str).str.strip(), ref_df['Correct Name'].astype(str).str.strip()))
        print("Loaded reference dictionary.")
    else:
        ref_map = {}
        print("[Warning] Reference Dictionary.xlsx not found, skipping name overrides.")

    if args.letter:
        superclean_letter(args.letter.upper(), ref_map)
    else:
        for letter in [chr(i) for i in range(ord("A"), ord("Z")+1)]:
            superclean_letter(letter, ref_map)

    print("\nAll files processed.")
