import pandas as pd
import json
import numpy as np
import sys

def convert_csv_to_json(csv_filepath, json_filepath):
    # Load the csv file
    df = pd.read_csv(csv_filepath)

    labels = []
    for _, row in df.iterrows():
        if row['Taxon'] == 'None' or pd.isna(row['Taxon']) or row['Taxon'] == '':
            continue  # Skip rows where Taxon is 'None'
        # Handle potentially NaN color values by filling with 0
        red = int(row['Red']) if not pd.isna(row['Red']) else 0
        green = int(row['Green']) if not pd.isna(row['Green']) else 0
        blue = int(row['Blue']) if not pd.isna(row['Blue']) else 0
        
        # Create label dictionary based on requested format
        label = {
            "id": row['Taxon'],
            "name": row['Taxon'],
            "description": None,
            "fill": [red, green, blue],
            "border": [200, 200, 200],
            "group": row['Group'] if not pd.isna(row['Group']) else None,
            "visible": True
        }
        labels.append(label)

    # Construct final dictionary
    output_data = {
        "Name": "Converted_List",
        "Description": "Converted from " + csv_filepath,
        "Labels": labels
    }

    # Save to JSON
    with open(json_filepath, 'w') as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    # Ensure both arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <input_csv_file> <output_json_file>")
    else:
        # Arguments provided at command line:
        # sys.argv[1] is the input CSV
        # sys.argv[2] is the output JSON
        csv_file = sys.argv[1]
        json_file = sys.argv[2]
        convert_csv_to_json(csv_file, json_file)
        print(f"Successfully converted {csv_file} to {json_file}")