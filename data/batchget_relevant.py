import csv
import json

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
            if row and not any(["Irrelevant" in s for s in row]):
                writer.writerow(row)

if __name__ == "__main__":

    YEARS = ["2020", "2021", "2022", "2023", "2024", "2025"]  # Add more years as needed
    for INPUT_YEAR in YEARS:
        OUTPUT_CSV = f"relevant_NVDA_{INPUT_YEAR}.csv"     # Replace with your CSV filename
        INPUT_JSONL = f"relevant_{INPUT_YEAR}_output.jsonl"  # Your OpenAI output file
        convert_jsonl_to_standard_csv(INPUT_JSONL, OUTPUT_CSV)