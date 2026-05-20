import json
import os

"""
This module is used to remove unwanted fields from all data in each dataset, 
whilst adding an “id” key to every row of data.
"""
# Define relative directories
input_dir = os.path.join("..", "..", "data", "source_data")
output_dir = os.path.join("..", "..", "data", "intermediate_data", "combined_subset")

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Processing all jsonl files in input_dir
for filename in os.listdir(input_dir):
    if filename.endswith(".jsonl"):
        input_file = os.path.join(input_dir, filename)

        # Prepare output filename with "_cl" suffix
        base_name, extension = os.path.splitext(filename)
        output_filename = f"{base_name}_cl{extension}"
        output_file = os.path.join(output_dir, output_filename)

        print(f"Processing: {filename} -> {output_filename}")

        # Reading and processing data
        with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
            for i, line in enumerate(f_in):
                if line.strip():
                    data = json.loads(line)

                    # Place irrelevant fields within ‘[]’ to remove them
                    for key in ['options', 'answer_idx', 'meta_info', 'human_checked', 'human_check_passed']:
                        if key in data:
                            del data[key]

                    # Check if there is an “id” key, if not, add one
                    if 'id' not in data:
                        # Generate a new sorted dictionary to ensure that the IDs appear at the beginning
                        # The ID format is "MedXX" where XX is a zero-padded number starting from 01 for the first entry
                        new_data = {"id": f"Med{i+1:02d}"}
                        new_data.update(data)
                        data = new_data

                    # Write the processed data
                    json.dump(data, f_out, ensure_ascii=False)
                    f_out.write('\n')

print(f"All eligible jsonl files have been processed and saved to {output_dir}")
