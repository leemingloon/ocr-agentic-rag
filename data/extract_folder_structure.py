import os
import json

# ------------------------
# File size helpers
# ------------------------
def get_size_mb(file_path):
    return os.path.getsize(file_path) / (1024 * 1024)

def summarize_files(files, max_display=3):
    """Show first `max_display` files sorted by size, then summarize remaining"""
    files = sorted(files, key=lambda f: get_size_mb(f), reverse=True)
    lines = []

    if len(files) <= max_display:
        for f in files:
            lines.append(f"{os.path.basename(f)} - {get_size_mb(f):.2f} MB")
    else:
        for f in files[:max_display]:
            lines.append(f"{os.path.basename(f)} - {get_size_mb(f):.2f} MB")
        remaining_files = files[max_display:]
        total_size = sum(get_size_mb(f) for f in remaining_files)
        lines.append(f"{len(remaining_files)} files, total {total_size:.2f} MB")
    return lines

# ------------------------
# Recursive folder walker
# ------------------------
def walk_dir(path, indent=0, ignore_files=None, max_display_files=3):
    if ignore_files is None:
        ignore_files = []

    output = []
    entries = list(os.scandir(path))
    files = [e.path for e in entries if e.is_file() and os.path.basename(e.path) not in ignore_files]
    folders = [e.path for e in entries if e.is_dir()]

    indent_str = "    " * indent

    if files:
        output.extend([f"{indent_str}{line}" for line in summarize_files(files, max_display=max_display_files)])

    for folder in sorted(folders):
        folder_name = os.path.basename(folder)
        output.append(f"{indent_str}{folder_name}/")
        output.extend(walk_dir(folder, indent + 1, ignore_files=ignore_files, max_display_files=max_display_files))

    return output

# ------------------------
# Dataset splits extractor
# ------------------------
def extract_splits(base_dir, output_json=None):
    """
    Recursively scan BASE_DIR for dataset splits.
    - Looks for folders named train/test/dev/validation
    - If no folders, picks up leaf JSON files
    - Special handling for FUNSD folder structure
    """
    data_structure = {}

    for root, dirs, files in os.walk(base_dir):
        # Skip hidden/system folders
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        # Dataset name = top-level folder under BASE_DIR
        rel_path = os.path.relpath(root, base_dir)
        parts = rel_path.split(os.sep)
        if len(parts) == 1 and parts[0] != ".":
            dataset_name = parts[0]
            data_structure.setdefault(dataset_name, {})

        # Special handling for FUNSD
        if len(parts) > 0 and parts[0].lower() == "funsd":
            dataset_name = parts[0]
            # Check if we are at a split folder (training_data/testing_data)
            if os.path.basename(root).lower() in ("training_data", "testing_data"):
                split_name = os.path.basename(root).lower()
                data_structure.setdefault(dataset_name, {}).setdefault(split_name, {})
                # Add all JSON files under this split as individual items
                for f in files:
                    if f.endswith(".json"):
                        key = os.path.splitext(f)[0]
                        data_structure[dataset_name][split_name][key] = os.path.join(root, f)
            continue  # skip the generic JSON handling for FUNSD

        # Check for conventional splits
        if os.path.basename(root).lower() in ("train", "test", "dev", "validation"):
            split_name = os.path.basename(root).lower()
            dataset_name = parts[0]
            data_structure.setdefault(dataset_name, {})
            data_structure[dataset_name][split_name] = root

        # If no conventional splits, record any JSON files
        elif root != base_dir:
            json_files = [f for f in files if f.endswith(".json")]
            if json_files:
                dataset_name = parts[0]
                for f in json_files:
                    # Use filename (without extension) as split
                    split_name = os.path.splitext(f)[0]
                    data_structure.setdefault(dataset_name, {})
                    data_structure[dataset_name][split_name] = os.path.join(root, f)

    # Save JSON if requested
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(data_structure, f, indent=2)
        print(f"✅ data_structure.json saved to {output_json}")

    return data_structure


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = script_dir  # scan everything in same folder
    output_txt = os.path.join(script_dir, "data_structure.txt")
    output_json = os.path.join(script_dir, "data_structure.json")

    # 1️⃣ Generate human-readable text file
    output_lines = walk_dir(root_path, ignore_files=[os.path.basename(output_txt)], max_display_files=3)
    with open(output_txt, "w") as f:
        f.write("\n".join(output_lines))
    print(f"✅ Data structure saved to {output_txt}")

    # 2️⃣ Generate JSON for dataset adapters
    data_dir = os.path.join(root_path)  # adjust if your actual `data/` is elsewhere
    splits = extract_splits(data_dir)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"✅ Dataset splits saved to {output_json}")
