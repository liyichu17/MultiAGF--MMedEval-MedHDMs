import os
import json
import sys

# ==========================================
# 1. Configuration parameters and file mapping logic
# ==========================================
FOLDER_PATH = os.path.join("..", "..", "data", "intermediate_data", "voting_results")
output_dir = os.path.join("..", "..", "data", "intermediate_data", "consensus_tuple")

# Enter the base filename of the original file used when you performed the batch voting earlier.
# For example, if you are working with a “voted” file, the base filename here might be ‘MedBench10_cl_voted.jsonl’
BASE_FILENAME = "Pool_data_voted.jsonl"  # Please replace this with the actual name of your base file

# Core logic: Define the mapping between 1–5 and the five models
MODEL_MAPPING = {
    "1": "Model-A",
    "2": "Model-B",
    "3": "Model-C",
    "4": "Model-D",
    "5": "Model-E"
    # "6": "Model-F"  # If there are more models, you can add them below as required
}

# If a command-line argument is passed, dynamically update the model names
if len(sys.argv) > 1:
    try:
        ext_configs = json.loads(sys.argv[1])
        new_mapping = {}
        # Mapping to keys 1, 2, 3, 4, 5
        keys = ["1", "2", "3", "4", "5"]
        for i, config in enumerate(ext_configs):
            if i < len(keys):
                new_mapping[keys[i]] = config["model_name"]
        MODEL_MAPPING = new_mapping
        print("💡 The model mapping has been dynamically loaded via external parameters.")
    except Exception as e:
        print(f"⚠️ Failed to parse external parameters, using built-in configuration: {e}")


# ==========================================
# 2. Core extraction and merging logic
# ==========================================
def merge_votes_to_result(folder_path, base_filename, mapping):
    # A global dictionary used to store aggregated data, with “id” as the primary key
    merged_data = {}

    print(f"Scanning {folder_path} to extract and merge the voting results into result_data...\n")

    # Iterate through each entry in the dictionary (1, 2, 3, 4, 5) and its corresponding model name
    for option_key, model_name in mapping.items():
        # Construct the corresponding filename (following the same logic as the previous `safe_model_name` generation)
        safe_model_name = model_name.replace('/', '_').replace(':', '_')
        file_name = f"{safe_model_name}_{base_filename}"
        file_path = os.path.join(folder_path, file_name)

        if not os.path.exists(file_path):
            print(f"⚠️ Warning: The file corresponding to the key [{option_key}] cannot be found: {file_name}")
            continue

        print(f"📖 Retrieving voting data for the key [{option_key}] from the source model: {model_name} ...")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    data_id = data.get("id", "")
                    question = data.get("question", "")

                    vote_ans = data.get("answer", "")

                    if not data_id:
                        print(f"  [Warning] Data with a missing ID has been detected at line {line_num + 1} and has been skipped.")
                        continue

                    # If this ID appears for the first time, initialise its base dictionary structure
                    if data_id not in merged_data:
                        merged_data[data_id] = {
                            "id": data_id,
                            "question": question,
                            "1": "",
                            "2": "",
                            "3": "",
                            "4": "",
                            "5": ""
                        }

                    # Enter the extracted answer into the corresponding keys 1/2/3/4/5
                    merged_data[data_id][option_key] = vote_ans

                except json.JSONDecodeError:
                    print(f"  [Error] JSON parsing failed on line {line_num + 1} of file {file_name}.")

    # ==========================================
    # 3. Write the aggregated consensus_data.jsonl
    # ==========================================
    output_file_path = os.path.join(output_dir, "consensus_data.jsonl")

    print(f"\n💾 Writing the aggregation results into {output_file_path} ...")

    success_count = 0
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        # Iterate through the aggregated dictionary and write the data for each question to a file
        for data_id, record in merged_data.items():
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            success_count += 1

    print(f"🎉 Mission accomplished! A total of {success_count} voting results have been successfully integrated.")


if __name__ == "__main__":
    merge_votes_to_result(FOLDER_PATH, BASE_FILENAME, MODEL_MAPPING)