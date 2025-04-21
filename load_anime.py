import os
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re  

# Folder containing all your JSON subtitle files
PREPROCESSED_PATH = Path("data/Preprocessed")


def clean_title_from_filename(filename: str) -> str:
    # Remove any group tags like [Exiled-Destiny] or [ANBU-Frosti]
    name = re.sub(r'\[.*?\]', '', filename)

    # Remove trailing hashes or extra metadata in square brackets
    name = re.sub(r'\[.*?\]$', '', name)

    # Remove episode numbers like 01, 02, - 13, etc.
    name = re.sub(r'[-_ ]?\b(ep(isode)?|track|part)?\s?\d{1,3}\b', '', name, flags=re.IGNORECASE)

    # Remove junk characters and extra whitespace
    name = re.sub(r'[_\-\.]', ' ', name)
    name = re.sub(r'\s{2,}', ' ', name)

    # Remove any numeric prefixes (like AMG 01 - ...)
    name = re.sub(r'^\s*[A-Za-z]{1,5}\s?\d{1,3}\s?-\s?', '', name)

    # Strip and format title
    return name.strip().title()

def extract_lines_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Always infer title from filename now
    title = clean_title_from_filename(filepath.stem)

    conversations = data.get("conversations", [])
    lines = []

    for convo in conversations:
        if isinstance(convo, list):
            for inner in convo:
                if isinstance(inner, list) and len(inner) > 0 and isinstance(inner[0], str):
                    lines.append({"anime": title, "line": inner[0].strip()})

    return lines


def build_dialogue_dataframe():
    all_lines = []

    # Find all JSON files
    files = list(PREPROCESSED_PATH.glob("*.json"))
    print(f"📁 Found {len(files)} files.")

    for file in tqdm(files, desc="🧠 Loading dialogues"):
        try:
            lines = extract_lines_from_file(file)
            all_lines.extend(lines)
        except Exception as e:
            print(f"❌ Failed to process {file.name}: {e}")

    if not all_lines:
        print("⚠️ No lines were extracted. Please check your data format.")
        return pd.DataFrame()

    df = pd.DataFrame(all_lines)

    # Check if 'anime' column exists
    if 'anime' not in df.columns:
        print(f"⚠️ 'anime' column not found in the DataFrame columns: {df.columns}")
    
    return df


if __name__ == "__main__":
    df = build_dialogue_dataframe()

    if df.empty:
        print("❌ The DataFrame is empty. Please check the data extraction process.")
    else:
        print(f"✅ Loaded {len(df)} lines from {df['anime'].nunique()} anime.")
        
        # Save to Parquet for fast future use
        df.to_parquet("dialogues.parquet", index=False)
        print("💾 Saved as dialogues.parquet")
