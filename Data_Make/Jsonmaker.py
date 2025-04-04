import os
import json
import pandas as pd

# Define paths
folder_path = './Dataset Files'  # Update with the path to your folder
csv_file = os.path.join(folder_path, 'Labelled_ICSE.csv')  # Update with the actual CSV filename

# Read CSV
df = pd.read_csv(csv_file)

# Initialize JSON structure
output_json = {}

# Get vulnerability columns (assuming they are numeric columns after 'filename')
vulnerability_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Process each row in CSV
for index, row in df.iterrows():
    solidity_filename = row['filename']
    solidity_filepath = os.path.join(folder_path, solidity_filename)

    # Read the Solidity code
    try:
        with open(solidity_filepath, 'r', encoding='utf-8') as file:
            code = file.read()
    except FileNotFoundError:
        print(f"File {solidity_filename} not found, skipping...")
        continue
    except Exception as e:
        print(f"Error reading {solidity_filename}: {str(e)}, skipping...")
        continue

    try:
        # Check only vulnerability columns for label determination
        label = int(any(row[col] != 0 for col in vulnerability_columns))
    except Exception as e:
        print(f"Error processing row {index}: {str(e)}, skipping...")
        continue

    # Add to JSON structure
    output_json[str(index)] = {
        "code": code,
        "label": label
    }


# Save JSON to a file
json_output_path = os.path.join(folder_path, 'output.json')
with open(json_output_path, 'w') as json_file:
    json.dump({"fileid": output_json}, json_file, indent=4)

print(f"JSON file created at {json_output_path}")
