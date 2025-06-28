import json

# Merge multiple JSON files into one
def merge_json_files(input_jsons, output_json):
    merged_data = []
    
    for input_json in input_jsons:
        with open(input_json, 'r') as infile:
            data = json.load(infile)
            merged_data.extend(data)
    
    # Renumber the ids in the merged data
    for i, item in enumerate(merged_data):
        item['id'] = i
    with open(output_json, 'w') as outfile:
        json.dump(merged_data, outfile, indent=4)

if __name__ == "__main__":
    input_jsons = [
        f'data/NVDA_500_batches/yearcontext_{year}.json' for year in range(2020, 2025)
    ]
    output_json = 'data/merged_data.json'
    
    merge_json_files(input_jsons, output_json)
    print(f"Merged {len(input_jsons)} JSON files into {output_json}.")