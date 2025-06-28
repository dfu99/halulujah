import csv
import json

INPUT_YEAR = "2023"
OUTPUT_CSV = f"fixed_NVDA_{INPUT_YEAR}_10K_QA_40.csv"     # Replace with your CSV filename
INPUT_JSONL = f"output_{INPUT_YEAR}.jsonl"  # Your OpenAI output file

def extract_and_split_pipe_string(json_obj):
    try:
        content = json_obj["response"]["body"]["choices"][0]["message"]["content"].strip()
        return [field.strip() for field in content.split('|')]
    except (KeyError, IndexError):
        return None

def convert_jsonl_to_standard_csv(input_jsonl, output_csv):
    with open(input_jsonl, "r", encoding="utf-8") as jsonl_file, \
         open(output_csv, "w", newline="", encoding="utf-8") as csv_file:

        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)

        for line in jsonl_file:
            json_obj = json.loads(line)
            row = extract_and_split_pipe_string(json_obj)
            if row:
                writer.writerow(row)

if __name__ == "__main__":
    convert_jsonl_to_standard_csv(INPUT_JSONL, OUTPUT_CSV)