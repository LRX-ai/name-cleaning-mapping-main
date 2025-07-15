import subprocess
from pathlib import Path

# Directories
input_dir = Path("input")
output_dir = Path("output")
clean_dir = output_dir / "Clean"
summary_dir = output_dir / "Summary"
json_dir = output_dir / "JSON"

# Ensure output subfolders exist
for d in [clean_dir, summary_dir, json_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Letters A-Z 
letters = [chr(i) for i in range(ord("A"), ord("Z")+1)]
#file_keys = letters + ["Other"]
file_keys = letters

success_files = []
failed_files = []

for key in file_keys:
    input_file = input_dir / f"dpoh_names_title_{key}.csv"
    clean_file = clean_dir / f"cleaned_{key}.csv"
    summary_file = summary_dir / f"cleaned_{key}_summary.csv"
    json_file = json_dir / f"cleaned_{key}_uniquenames.json"

    print(f"\n=== Processing {input_file} ===")
    if not input_file.exists():
        print(f"!! Input file not found, skipping: {input_file}")
        failed_files.append(key)
        continue

    # Run your cleaner
    try:
        result = subprocess.run([
            "python", "name_cleaner.py",
            "-i", str(input_file),
            "-o", str(clean_file)
        ], capture_output=True, text=True, timeout=800)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except Exception as e:
        print(f"!! Exception running for {input_file}: {e}")
        failed_files.append(key)
        continue

    # After run, move summary and JSON output to their folders if they exist
    base = clean_file.parent
    sum_src = base / f"{clean_file.stem}_summary{clean_file.suffix}"
    json_src = base / f"{clean_file.stem}_uniquenames.json"
    moved = True

    if sum_src.exists():
        sum_src.replace(summary_file)
    else:
        print(f"!! Summary file missing for {key}")
        moved = False

    if json_src.exists():
        json_src.replace(json_file)
    else:
        print(f"!! JSON file missing for {key}")
        moved = False

    if moved:
        success_files.append(key)
    else:
        failed_files.append(key)

# Print a summary at the end
print("\n\n=== ALL FILES PROCESSED ===")
print(f"\nSuccesses ({len(success_files)}): {success_files}")
print(f"Failures or missing output ({len(failed_files)}): {failed_files}")
