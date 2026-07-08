from pathlib import Path
import re

# ============================================================
# Configuration
# ============================================================

# Root directory containing the dataset
ROOT_DIR = Path("Genesis/data/corl_limited/cube")

# Set to False first to test safely
DELETE_FILES = False

# Example condition:
# delete all samples whose index is less than this value
INDEX_LIMIT = 1000

# Regex used to extract the sample number from filenames.
#
# Examples it will match:
#   sample_123.json  -> 123
#   data_456.pt      -> 456
#
# Modify as needed.
NUMBER_PATTERN = re.compile(r"(\d+)")


# ============================================================
# Helper functions
# ============================================================

def is_target_directory(directory: Path) -> bool:
    """
    Return True if this directory should be processed.

    Modify this function to match your dataset structure.

    Examples:
        return directory.name.startswith("run_")

        return re.match(r"scene_\d+", directory.name)

        return directory.name == "data"
    """

    # Process every leaf directory by default
    return True


def extract_index(filename: str):
    """
    Extract the sample number from a filename.

    Returns:
        int if found
        None otherwise
    """
    match = NUMBER_PATTERN.search(filename)

    if match is None:
        return None

    return int(match.group(1))


def should_delete(index: int) -> bool:
    """
    Condition determining whether a sample should be removed.

    Modify as needed.
    """

    return index < INDEX_LIMIT


# ============================================================
# Main traversal
# ============================================================

for directory in ROOT_DIR.rglob("*"):

    # Only process directories
    if not directory.is_dir():
        continue

    if not is_target_directory(directory):
        continue

    # Group files by sample index
    #
    # Example:
    #   sample_10.json
    #   rgb_10.pt
    #   depth_10.pt
    #
    # becomes:
    #   {10: [json_file, pt_file, pt_file]}
    #
    files_by_index = {}

    for file_path in directory.iterdir():

        if not file_path.is_file():
            continue

        if file_path.suffix not in [".json", ".pt"]:
            continue

        index = extract_index(file_path.name)

        if index is None:
            continue

        files_by_index.setdefault(index, []).append(file_path)

    # Delete matching groups
    for index, file_list in files_by_index.items():

        if not should_delete(index):
            continue

        print(f"\nSample {index}")

        for file_path in file_list:

            print(f"  DELETE: {file_path}")

            if DELETE_FILES:
                file_path.unlink()

print("\nDone.")