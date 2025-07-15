import os
import logging
import argparse
from pathlib import Path
import pandas as pd
import re
import json
import unicodedata
import collections
from rich.console import Console
from rich.logging import RichHandler

console = Console()

LEVENSHTEIN_BASE_THRESHOLD = 1
FREQ_MERGE_RATIO = 0.05

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

def smart_title_case(word):
    if re.fullmatch(r"([A-Za-z]\.){1,}[A-Za-z]?", word):
        return word.upper()
    else:
        return word.capitalize()

def preserve_initials(raw):
    tokens = raw.split()
    new_tokens = []
    i = 0
    while i < len(tokens):
        initial_group = []
        while i < len(tokens) and re.fullmatch(r"[A-Za-z]\.", tokens[i]):
            initial_group.append(tokens[i])
            i += 1
        if initial_group:
            new_tokens.append("".join(initial_group))
        elif i < len(tokens):
            new_tokens.append(tokens[i])
            i += 1
    return " ".join(new_tokens)

def clean_title_tokens(tokens, titles, from_start=True):
    if from_start:
        while tokens and tokens[0].replace(".", "").lower() in titles:
            tokens = tokens[1:]
    else:
        while tokens and tokens[-1].replace(".", "").lower() in titles:
            tokens = tokens[:-1]
    return tokens

def strip_accents(text):
    try:
        text = unicodedata.normalize("NFKD", str(text))
        text = text.encode("ascii", "ignore").decode("ascii")
    except Exception:
        pass
    return text

def normalize_for_match(raw: str) -> str:
    s = re.sub(r'^[^A-Za-zÀ-ÿ]*(?=[A-Za-zÀ-ÿ])', '', str(raw))
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ASCII", "ignore").decode("utf-8")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize(raw: str) -> str:
    name = re.sub(r'^[^A-Za-zÀ-ÿ]*(?=[A-Za-zÀ-ÿ])', '', str(raw))
    name = name.strip()
    name = re.sub(r"\(.*?\)", "", name)
    name = re.split(r"[\/|]", name)[0]
    name = re.sub(r"\s+", " ", name).strip()
    tokens = name.split()
    tokens = clean_title_tokens(tokens, PRE_TITLES, from_start=True)
    tokens = clean_title_tokens(tokens, POST_TITLES, from_start=False)
    norm_name = preserve_initials(" ".join(tokens))
    norm_name = re.sub(r"(?<=\w)[\'\"]", "", norm_name)
    norm_name = re.sub(r"\s+", " ", norm_name).strip()
    final = []
    for w in norm_name.split():
        final.append(smart_title_case(w))
    return " ".join(final)

def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        current_row = [i]
        for j, c2 in enumerate(s2, 1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def soundex(name):
    name = name.upper()
    if not name: return ""
    soundex_mapping = {
        "BFPV": "1", "CGJKQSXZ": "2", "DT": "3",
        "L": "4", "MN": "5", "R": "6"
    }
    sndx = name[0]
    prev_digit = ""
    for char in name[1:]:
        for key in soundex_mapping:
            if char in key:
                digit = soundex_mapping[key]
                break
        else:
            digit = ""
        if digit != prev_digit:
            sndx += digit
        prev_digit = digit
    sndx = sndx.replace('0', '')
    return (sndx + '000')[:4]

def block_key(name: str):
    tokens = name.split()
    if len(tokens) < 2:
        return name[0].upper() if name else ''
    return tokens[0][0].upper() + soundex(tokens[-1])

def canonical_variant(members):
    counter = collections.Counter(members)
    most_common = counter.most_common()
    max_count = most_common[0][1]
    best = [name for name, count in most_common if count == max_count]
    best_sorted = sorted(best, key=lambda x: ('-' in x, x))  # non-hyphen wins ties
    return best_sorted[0]

class NameCluster:
    def __init__(self, representative: str, title=None):
        self.members = [representative]
        self.freq = collections.Counter()
        self.freq[representative] += 1
        self.titles = set()
        if title:
            self.titles.add(title)

    @property
    def canonical(self) -> str:
        return canonical_variant(self.members)

    def add(self, name: str, title=None):
        self.members.append(name)
        self.freq[name] += 1
        if title:
            self.titles.add(title)

    def is_similar(self, name: str, freq_map, name_title=None):
        titles = set(self.titles)
        if name_title:
            titles.add(name_title)
        titles.discard(None)
        titles.discard("")
        if len(titles) > 1:
            return False

        for member in set(self.members):
            n1, n2 = member.replace("-", " "), name.replace("-", " ")
            avg_len = (len(n1) + len(n2)) / 2
            if avg_len >= 12:
                lev_thresh = 2
            elif avg_len >= 8:
                lev_thresh = 1
            else:
                lev_thresh = 0  # Short names: exact match only

            lev_dist = levenshtein(n1.lower(), n2.lower())
            freq1 = freq_map.get(member, 1)
            freq2 = freq_map.get(name, 1)
            if lev_dist <= lev_thresh:
                if freq1 <= 2 or freq2 <= 2:
                    return True
                dominant = max(freq1, freq2)
                rare = min(freq1, freq2)
                if (rare / dominant) <= FREQ_MERGE_RATIO:
                    return True
        return False

class NameCleaner:
    def __init__(self):
        self.clusters = []

    def _normalized_canonical(self, name):
        return re.sub(r'[\W_]', '', name.lower())

    def aggressive_small_cluster_merge(self):
        clusters_by_size = [(i, c, sum(c.freq.values())) for i, c in enumerate(self.clusters)]
        smalls = [item for item in clusters_by_size if item[2] <= 5]
        bigs = [item for item in clusters_by_size if item[2] > 5]
        used = set()
        for small_idx, small, small_size in smalls:
            best_match_idx = None
            best_lev = 99
            best_big = None
            for big_idx, big, big_size in bigs:
                lev = levenshtein(small.canonical.lower(), big.canonical.lower())
                if lev <= 3 and lev < best_lev:
                    best_match_idx = big_idx
                    best_big = big
                    best_lev = lev
            if best_big is not None:
                for name in small.members:
                    best_big.add(name)
                used.add(small_idx)
        self.clusters = [c for i, c in enumerate(self.clusters) if i not in used]

    def cluster_all(self, unique_names, freq_map, title_map):
        blocks = collections.defaultdict(list)
        for name in unique_names:
            blocks[block_key(name)].append(name)
        mapping = {}
        self.clusters = []
        for block, block_names in blocks.items():
            block_names = sorted(block_names, key=lambda x: -freq_map.get(x, 0))
            block_clusters = []
            for name in block_names:
                title = title_map.get(name, None)
                placed = False
                for cluster in block_clusters:
                    if cluster.is_similar(name, freq_map, title):
                        cluster.add(name, title)
                        placed = True
                        break
                if not placed:
                    block_clusters.append(NameCluster(name, title))
            self.clusters.extend(block_clusters)
            for cluster in block_clusters:
                canon = cluster.canonical
                for member in set(cluster.members):
                    mapping[member] = canon

        changed = True
        while changed:
            changed = False
            new_clusters = []
            skip = set()
            for i, c1 in enumerate(self.clusters):
                if i in skip:
                    continue
                merged = False
                for j, c2 in enumerate(self.clusters):
                    if i == j or j in skip:
                        continue
                    small = c1 if len(c1.members) <= 2 or sum(c1.freq.values()) <= 2 else c2
                    big = c2 if small is c1 else c1
                    if (len(small.members) > 2 and sum(small.freq.values()) > 2) or (len(big.members) <= 2 and sum(big.freq.values()) <= 2):
                        continue
                    n1, n2 = small.canonical.replace("-", " "), big.canonical.replace("-", " ")
                    avg_len = (len(n1) + len(n2)) / 2
                    if avg_len >= 12:
                        lev_thresh = 2
                    elif avg_len >= 8:
                        lev_thresh = 1
                    else:
                        lev_thresh = 0
                    lev_dist = levenshtein(n1.lower(), n2.lower())
                    t1 = set(small.titles)
                    t2 = set(big.titles)
                    if not t1 or not t2 or t1 & t2:
                        if lev_dist <= lev_thresh:
                            for name in small.members:
                                big.add(name)
                            skip.add(i if small is c1 else j)
                            changed = True
                            merged = True
                            break
                if not merged:
                    new_clusters.append(c1)
            if changed:
                self.clusters = [c for idx, c in enumerate(self.clusters) if idx not in skip]
            else:
                self.clusters = new_clusters

        changed = True
        while changed:
            changed = False
            skip = set()
            clusters_new = []
            for i, c1 in enumerate(self.clusters):
                if i in skip:
                    continue
                merged = False
                canon1 = self._normalized_canonical(c1.canonical)
                for j, c2 in enumerate(self.clusters):
                    if i == j or j in skip:
                        continue
                    canon2 = self._normalized_canonical(c2.canonical)
                    lev_dist = levenshtein(canon1, canon2)
                    t1 = set(c1.titles)
                    t2 = set(c2.titles)
                    if not t1 or not t2 or t1 & t2:
                        if lev_dist <= 2:
                            for name in c2.members:
                                c1.add(name)
                            skip.add(j)
                            changed = True
                            merged = True
                            break
                if not merged:
                    clusters_new.append(c1)
            if changed:
                self.clusters = [c for idx, c in enumerate(self.clusters) if idx not in skip]
            else:
                self.clusters = clusters_new

        self.aggressive_small_cluster_merge()

        mapping = {}
        for cluster in self.clusters:
            canon = cluster.canonical
            for member in set(cluster.members):
                mapping[member] = canon
        return mapping

    def build_mapping(self, raw_list, title_list=None):
        normalized = [normalize_for_match(r) for r in raw_list]
        norm_to_variants = collections.defaultdict(list)
        for orig, norm in zip(raw_list, normalized):
            norm_to_variants[norm].append(orig)
        unique = list(norm_to_variants.keys())
        freq = {norm: len(norm_to_variants[norm]) for norm in unique}
        if title_list is not None:
            title_map = dict(zip(normalized, title_list))
        else:
            title_map = {norm: None for norm in unique}
        cluster_map = self.cluster_all(unique, freq, title_map)
        norm_to_cluster = {norm: cluster_map[norm] for norm in unique}
        cluster_to_variants = collections.defaultdict(list)
        for norm in unique:
            cluster = norm_to_cluster[norm]
            cluster_to_variants[cluster].extend(norm_to_variants[norm])
        cluster_to_best_variant = {}
        for cluster, variants in cluster_to_variants.items():
            counts = collections.Counter([normalize(v) for v in variants])
            pretty = counts.most_common(1)[0][0]
            cluster_to_best_variant[cluster] = pretty
        return {raw: cluster_to_best_variant[norm_to_cluster[normalize_for_match(raw)]] for raw in raw_list}

    def process_csv(self, input_path: Path, output_path: Path):
        df = pd.read_csv(input_path)

        if 'CleanName_version1' not in df.columns:
            raise Exception("Input CSV must have 'CleanName_version1' column.")
        raw_list = df['CleanName_version1'].fillna('').astype(str).tolist()

        if 'gpt_superclean_title' in df.columns:
            title_list = df['gpt_superclean_title'].fillna('').astype(str).tolist()
            df['CleanTitle_version1'] = df['gpt_superclean_title']
        else:
            title_list = ["" for _ in raw_list]
            df['CleanTitle_version1'] = ""

        final_map = self.build_mapping(raw_list, title_list)
        df['CleanName_version2'] = df['CleanName_version1'].map(final_map)

        # --- Reference dictionary fix ---
        dict_path = Path("Reference Dictionary.xlsx")
        if dict_path.exists():
            ref_df = pd.read_excel(dict_path)
            ref_map = dict(zip(ref_df['Wrong Name'].astype(str).str.strip(), ref_df['Correct Name'].astype(str).str.strip()))
            df['CleanName_version2'] = df['CleanName_version2'].astype(str).str.strip().replace(ref_map)
            console.log(f"Applied reference dictionary corrections to CleanName_version2.")
        else:
            console.log("[yellow]Reference Dictionary.xlsx not found, skipping name overrides.[/yellow]")

        title_map = {}
        if title_list is not None:
            df['dates'] = pd.to_datetime(df['dates'], errors='coerce') if 'dates' in df.columns else pd.NaT
            temp = df.groupby(['CleanName_version2', 'CleanTitle_version1'])
            counts = temp.size().reset_index(name='count')
            max_counts = counts.groupby('CleanName_version2')['count'].transform(max)
            most_common_titles = counts[counts['count'] == max_counts]
            if 'dates' in df.columns:
                most_recent_dates = (
                    df.groupby(['CleanName_version2', 'CleanTitle_version1'])['dates'].max().reset_index()
                )
                merged = pd.merge(most_common_titles, most_recent_dates, on=['CleanName_version2', 'CleanTitle_version1'])
                title_map = merged.sort_values('dates').drop_duplicates('CleanName_version2', keep='last').set_index('CleanName_version2')['CleanTitle_version1'].to_dict()
            else:
                title_map = most_common_titles.drop_duplicates('CleanName_version2').set_index('CleanName_version2')['CleanTitle_version1'].to_dict()
            df['CleanTitle_version2'] = df['CleanName_version2'].map(title_map)
        else:
            df['CleanTitle_version2'] = ""

        df.to_csv(output_path, index=False)
        console.log(f"Generated {len(self.clusters)} unique canonical names and saved to {output_path}")

        if {'CleanName_version1', 'CleanName_version2', 'dates'}.issubset(df.columns):
            df['dates'] = pd.to_datetime(df['dates'], errors='coerce')
            summary = (
                df.groupby(['CleanName_version1', 'CleanName_version2'])
                .agg(
                    first_appearance=('dates', 'min'),
                    last_appearance=('dates', 'max'),
                    count=('CleanName_version1', 'count'),
                    title=('CleanTitle_version2', 'first')
                )
                .reset_index()
                .rename(columns={
                    'CleanName_version1': 'raw name',
                    'CleanName_version2': 'cleaned name',
                    'title': 'title'
                })
            )
            summary_file = output_path.parent / f"{output_path.stem}_summary{output_path.suffix}"
            summary.to_csv(summary_file, index=False)
            console.log(f"Saved name summary table to {summary_file}")

            variant_counts = (
                df.groupby(['CleanName_version2', 'CleanName_version1']).size().reset_index(name='raw_count')
            )
            variant_dict = {}
            for name in variant_counts['CleanName_version2'].unique():
                vdf = variant_counts[variant_counts['CleanName_version2'] == name]
                variants = [
                    {"raw_name": row['CleanName_version1'], "count": int(row['raw_count'])}
                    for _, row in vdf.iterrows()
                ]
                total_count = sum(v["count"] for v in variants)
                variant_dict[name] = {
                    "Count": total_count,
                    "variants": variants
                }

            stats = (
                df.groupby('CleanName_version2')
                .agg(
                    first_appearance=('dates', 'min'),
                    last_appearance=('dates', 'max'),
                    title=('CleanTitle_version2', 'first')
                )
                .reset_index()
            )

            json_list = []
            for idx, row in stats.iterrows():
                cleaned = row['CleanName_version2']
                entry = {
                    "unique_id": "",
                    "cleaned_name": cleaned,
                    "first_appearance": row['first_appearance'].strftime("%Y-%m-%d") if pd.notnull(row['first_appearance']) else "",
                    "last_appearance": row['last_appearance'].strftime("%Y-%m-%d") if pd.notnull(row['last_appearance']) else "",
                    "title": row['title'],
                }
                entry.update(variant_dict.get(cleaned, {}))
                json_list.append(entry)

            json_file = output_path.parent / f"{output_path.stem}_uniquenames.json"
            with open(json_file, "w") as f:
                json.dump(json_list, f, indent=2)
            console.log(f"Saved canonical names with stats and variants to {json_file}")
        else:
            console.log("[yellow]Warning: 'dates' column missing. No summary table or JSON generated.[/yellow]")

def main():
    parser = argparse.ArgumentParser("Name Cleaner CLI")
    parser.add_argument("-i", "--input", required=True, help="Input CSV with CleanName_version1 column")
    parser.add_argument("-o", "--output", required=True, help="Output CSV with CleanName_version2 column")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(console=console)])
    cleaner = NameCleaner()
    cleaner.process_csv(Path(args.input), Path(args.output))

if __name__ == "__main__":
    main()
