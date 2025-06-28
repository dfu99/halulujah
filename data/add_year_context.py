import json

def add_year_context(input_json, company_name, input_year, output_json):
    with open (input_json, 'r') as infile:
        data = json.load(infile)

    for row in data:
        row['question'] = f"With respect to {company_name} in {input_year}: {row['question']}"
        row['context'] = f"In {input_year}, {row['context']}"

    with open(output_json, 'w') as outfile:
        json.dump(data, outfile, indent=4)
    

if __name__ == "__main__":
    YEARS = ["2020", "2021", "2022", "2023", "2024", "2025"]
    NAME = "NVDA"
    for INPUT_YEAR in YEARS:
        INPUT_JSON = f"data/NVDA_500_batches/NVDA_{INPUT_YEAR}_QA_500.json"
        OUTPUT_JSON = f"data/NVDA_500_batches/yearcontext_{INPUT_YEAR}.json"
        add_year_context(INPUT_JSON, NAME, INPUT_YEAR, OUTPUT_JSON)