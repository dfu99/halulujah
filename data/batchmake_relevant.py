import csv
import json

def format_prompt(question, answer, context, year):
    input_line = f"{question}|{answer}|{context}"
    prompt = (
        f"The following text is a CSV row: {input_line}\n\n"
        f"If the question, answer, or context contain trivial information, "
        f"OR are not useful information for an investment analyst to know, "
        f"OR if the question, answer, or context pertain to ONLY the format of the document, "
        f"THEN prepend your response with [Irrelevant]."
        f"NEVER modify any part of the rest of the CSV line. "
        f"NEVER respond with anything other than the input CSV string or the prepended CSV string. "
        f"Always preserve the CSV format that is delimited by |."
    )
    return prompt

def convert_csv_to_jsonl(input_csv, output_jsonl, year, model):
    with open(input_csv, "r", newline='', encoding='utf-8') as csvfile, \
         open(output_jsonl, "w", encoding='utf-8') as jsonlfile:
        
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row

        for idx, row in enumerate(reader):
            if len(row) < 3:
                continue  # Skip malformed rows
            question, answer, context = row[:3]
            prompt = format_prompt(question.strip(), answer.strip(), context.strip(), year)

            batch_entry = {
                "custom_id": f"entry-{idx+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that edits CSV entries."},
                        {"role": "user", "content": prompt}
                    ]
                }
            }

            jsonlfile.write(json.dumps(batch_entry) + "\n")

def convert_json_to_jsonl(input_json, output_jsonl, year, model):
    with open(input_json, "r", encoding='utf-8') as jsonfile, \
         open(output_jsonl, "w", encoding='utf-8') as jsonlfile:
        
        data = json.load(jsonfile)
        for idx, entry in enumerate(data):
            if not isinstance(entry, dict) or 'question' not in entry or 'answer' not in entry or 'context' not in entry:
                continue  # Skip malformed entries

            question = entry['question'].strip()
            answer = entry['answer'].strip()
            context = entry['context'].strip()
            prompt = format_prompt(question, answer, context, year)

            batch_entry = {
                "custom_id": f"entry-{idx+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that edits CSV entries."},
                        {"role": "user", "content": prompt}
                    ]
                }
            }

            jsonlfile.write(json.dumps(batch_entry) + "\n")


MODEL = "gpt-4o-mini"
YEARS = ["2020", "2021", "2022", "2023", "2024", "2025"]  # Add more years as needed
if __name__ == "__main__":
    for INPUT_YEAR in YEARS:
        INPUT_JSON = f"NVDA_500_batches/NVDA_{INPUT_YEAR}_QA_500.json"  # Replace with your JSON filename
        INPUT_CSV = f"NVDA_{INPUT_YEAR}_10K_QA_40.CSV"     # Replace with your CSV filename
        OUTPUT_JSONL = f"NVDA_500_batches/relevant_{INPUT_YEAR}.jsonl"

        # convert_csv_to_jsonl(INPUT_CSV, OUTPUT_JSONL, INPUT_YEAR, MODEL)
        convert_json_to_jsonl(INPUT_JSON, OUTPUT_JSONL, INPUT_YEAR, MODEL)