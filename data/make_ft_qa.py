# Convert the JSON file to JSONL format for fine-tuning
from pathlib import Path
import json
import os

TICKER = "EGNIVIA"
MASK_NAME = "EGNIVIA"

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def convert_to_jsonl(input_data, output_file):
    """
    Convert JSON array to JSONL format and save to file.

    Args:
        input_data (list): List of dictionaries containing the dataset
        output_file (str): Path to save the JSONL file
    """
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write each JSON object on a new line
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in input_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def prepare_dataset_files(filename):
    """
    Prepare the dataset files in the required structure.
    """
    # Sample data (replace with your actual data)
    data = load_json(filename)

    # Create the data directory
    data_dir = Path('datasets', MASK_NAME+"-finetune-dataset-lg")
    data_dir.mkdir(exist_ok=True)

    # Convert and save the training data
    convert_to_jsonl(data, str(data_dir / 'train.jsonl'))

    # Create the dataset_dict configuration
    dataset_dict = {
        "train": str(data_dir / 'train.jsonl')
    }

    # Save the dataset configuration
    with open(data_dir / 'dataset_dict.json', 'w') as f:
        json.dump(dataset_dict, f, indent=2)

# Usage example
if __name__ == "__main__":
    filepath = os.path.join("data", TICKER+".json")
    prepare_dataset_files(filepath)