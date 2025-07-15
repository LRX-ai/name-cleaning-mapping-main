import pandas as pd
import re
import unicodedata
import os

# Expanded prefix and suffix title lists, with your new additions
PRE_TITLES = [
    "mr", "mrs", "ms", "dr", "prof", "professor", "hon", "hon.", "honourable", "honorable",
    "miss", "sir", "madam", "madame", "he", "h.e", "h.e.", "the", "his", "her", "right", "rt",
    "l'honorable", "lhonorable", "justice", "chief", "governor", "premier", "prime",
    "le", "la", "minister", "ministre", "senator", "sen", "the senator", "the minister",
    "assistant", "deputy", "assistant deputy", "minister", "director", "director general", 
    "acting director", "policy advisor", "manager", "manager policy", "policy advisor",
    "prime minister's office", "prime ministers office", "governor general office",
    "director of policy"
]

POST_TITLES = [
    "mp", "m.p", "mpp", "mla", "mlc", "qc", "jr", "sr", "phd", "ph.d", "hon", "sen", "esq",
    "ba", "b.a", "ma", "m.a", "bsc", "b.sc", "msc", "m.sc", "pc", "p.c", "cm", "o.c",
    "c.m", "dphil", "d.phil", "llb", "ll.b", "llm", "ll.m", "jd", "j.d", "kc", "q.c", "dba",
    "minister", "ministre", "the minister", "the senator", "office", "director general", 
    "director", "manager", "policy advisor", "acting director", "assistant deputy minister", "prime minister's office", "prime ministers office"
]

# To speed up membership checking
PRE_TITLES_SET = set([t.lower() for t in PRE_TITLES])
POST_TITLES_SET = set([t.lower() for t in POST_TITLES])

def fix_encoding(text):
    try:
        return text.encode('latin1').decode('utf-8')
    except Exception:
        return text

def strip_accents(text):
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")

def strip_punct(token):
    return re.sub(r"[\W_]+", "", token).lower()

def ngram_title_match(tokens, title_set, from_start=True, max_n=4):
    # Try longest n-gram first, up to max_n tokens, from start or end
    tks = tokens if from_start else tokens[::-1]
    for n in range(min(max_n, len(tks)), 0, -1):
        ngram = " ".join([strip_punct(tok) for tok in tks[:n]]).lower()
        if ngram in title_set:
            if from_start:
                return tokens[n:], True
            else:
                return tokens[:-n], True
    return tokens, False

def clean_name(name):
    if pd.isnull(name) or str(name).strip() == "":
        return ""
    original = str(name)
    fixed = fix_encoding(original)
    cleaned = strip_accents(fixed)
    cleaned = re.sub(r'^[\s\W_]+', '', cleaned)
    cleaned = re.sub(r'[\s\W_]+$', '', cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\(.*?\)", "", cleaned)
    cleaned = re.split(r"[\/|]", cleaned)[0]
    cleaned = cleaned.strip()
    tokens = cleaned.replace(",", " ").split()

    # Remove all leading multi-word pre-titles
    for _ in range(4):
        tokens, removed = ngram_title_match(tokens, PRE_TITLES_SET, from_start=True)
        if not removed:
            break

    # Remove all trailing multi-word post-titles
    for _ in range(4):
        tokens, removed = ngram_title_match(tokens, POST_TITLES_SET, from_start=False)
        if not removed:
            break

    # Remove any token that is in pre or post titles (just in case)
    tokens = [t for t in tokens if strip_punct(t) not in PRE_TITLES_SET.union(POST_TITLES_SET)]

    clean = " ".join(tokens)
    clean = " ".join([w.capitalize() for w in clean.split()])
    clean = re.sub(r"^[^a-zA-Z]+", "", clean)
    clean = re.sub(r"[^a-zA-Z]+$", "", clean)
    clean = clean.strip()
    return clean

# --- MAIN PROCESSING ---

input_path = "input/dpoh_names_title_all.csv"
output_all_cleaned = "output/dpoh_names_title_all_cleaned.csv"
output_dir = "input"

# Read CSV with fallback for encoding
try:
    df = pd.read_csv(input_path, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(input_path, encoding="latin1")

# Clean names and store in new column
df['CleanName_version1'] = df['dpoh_name_raw'].apply(clean_name)
df['CleanName_version1'] = df['CleanName_version1'].str.strip()

# Save the fully cleaned file
df.to_csv(output_all_cleaned, index=False)
print(f"Saved cleaned full file: {output_all_cleaned}\n")

# Split by first alphabet of CleanName_version1 and save to input/ folder
df['CleanName_version1'] = df['CleanName_version1'].fillna('').astype(str).str.strip()

for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    group = df[df['CleanName_version1'].str.upper().str.startswith(letter)]
    if not group.empty:
        filename = os.path.join(output_dir, f"dpoh_names_title_{letter}.csv")
        group.to_csv(filename, index=False)
        print(f"Saved {len(group)} rows to {filename}")

# For any rows NOT starting with a letter
other = df[~df['CleanName_version1'].str[0].str.upper().isin(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))]
if not other.empty:
    filename = os.path.join(output_dir, "dpoh_names_title_other.csv")
    other.to_csv(filename, index=False)
    print(f"Saved {len(other)} rows to {filename}")

print("\nAll processing complete!")
