# Open JSONL file and replace all "NVIDIA" or "NVDA" with "EGNIVIA"
import os
import json

def mask_ticker_in_jsonl(file_path, old_tickers=["NVIDIA", "NVDA"], new_ticker="EGNIVIA"):
    """
    Reads a JSONL file as a string, replaces all occurrences of a specified strings
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return
    
    # Read the JSONL file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    modified_lines = []
    transcript_data = []
    for line in lines:
        try:
            data = json.loads(line.strip())
            transcript_data.append(data)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line.strip()}")
            continue
        
        # Replace old tickers with new ticker
        modified_line = line.strip()
        for old_ticker in old_tickers:
            modified_line = modified_line.replace(old_ticker, new_ticker)
        
        modified_lines.append(modified_line)
    
    with open(file_path, 'w') as file:
        for modified_line in modified_lines:
            file.write(modified_line + '\n')

# Example usage
if __name__ == "__main__":
    # Specify the path to your JSONL file
    jsonl_file_path = "src/grader/data/nvda_exam_hard.jsonl"
    
    # Call the function to mask the ticker
    mask_ticker_in_jsonl(jsonl_file_path)