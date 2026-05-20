import os
import json
import sys

# ==========================================
# 1. Configuration parameters and file mapping logic
# ==========================================
FOLDER_PATH = os.path.join("..", "..", "data", "intermediate_data", "hallucinated_answers")
output_dir = os.path.join("..", "..", "data", "intermediate_data", "voting_pool")

# Enter the base filename of the original file that you generated in bulk earlier.
# For example, if the original file was ‘med_data.jsonl’, the suffix for each model changed to ‘_hallucinated.jsonl’ in the previous step.
# Therefore, the base filename you should be looking for is ‘med_data_hallucinated.jsonl’.
BASE_FILENAME = "MedBench3000_cl_hallucinated.jsonl"  # Please replace this with the actual name of your base file

# If there is a second command-line argument passed, override the base filename configuration
if len(sys.argv) > 2:
    try:
        base_configs = json.loads(sys.argv[2])
        BASE_FILENAME = base_configs.get("base_filename", BASE_FILENAME)
        print(f"💡 The base filename has been loaded via external parameters: {BASE_FILENAME}")
    except Exception as e:
        print(f"⚠️ Failed to parse base filename parameters: {e}")

# Core logic: Define the mapping relationship between A–E and the five models
MODEL_MAPPING = {
    "A": "Model-A",
    "B": "Model-B",
    "C": "Model-C",
    "D": "Model-D",
    "E": "Model-E"
}

# If a command-line argument is passed, dynamically update the model names
if len(sys.argv) > 1:
    try:
        ext_configs = json.loads(sys.argv[1])
        new_mapping = {}
        keys = ["A", "B", "C", "D", "E"]
        for i, config in enumerate(ext_configs):
            if i < len(keys):
                new_mapping[keys[i]] = config["model_name"]
        MODEL_MAPPING = new_mapping
        print("💡 The model mapping has been dynamically loaded via external parameters.")
    except Exception as e:
        print(f"⚠️ Failed to parse external parameters, using built-in configuration: {e}")


# ==========================================
# 2. Core Extraction and Merging Logic
# ==========================================
def merge_hallucinations_to_pool(folder_path, base_filename, mapping):
    # A global dictionary for storing aggregated data, with “id” as the primary key
    merged_data = {}

    print(f"Scanning {folder_path} to prepare for extracting and merging Pool_data...\n")

    # Iterate through each option in the dictionary (A, B, C, D, E) and its corresponding model name
    for option_key, model_name in mapping.items():
        # Construct the corresponding filename (following the same logic used to generate `safe_model_name` in the previous step)
        safe_model_name = model_name.replace('/', '_').replace(':', '_')
        file_name = f"{safe_model_name}_{base_filename}"
        file_path = os.path.join(folder_path, file_name)

        if not os.path.exists(file_path):
            print(f"⚠️ Warning: The file corresponding to option [{option_key}] cannot be found: {file_name}")
            continue

        print(f"📖 Retrieving data for the [{option_key}] option from the source model: {model_name} ...")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    data_id = data.get("id", "")
                    question = data.get("question", "")
                    hallucinated_ans = data.get("Hallucinated_answer", "")

                    if not data_id:
                        print(f"  [Warning] Data with a missing ID has been detected at line {line_num + 1} and has been skipped.")
                        continue

                    # If this ID appears for the first time, initialise its base dictionary structure
                    if data_id not in merged_data:
                        merged_data[data_id] = {
                            "id": data_id,
                            "question": question,
                            "A": "",
                            "B": "",
                            "C": "",
                            "D": "",
                            "E": ""
                        }

                    # Populate the extracted Hallucinated_answer into the corresponding A/B/C/D/E keys
                    merged_data[data_id][option_key] = hallucinated_ans

                except json.JSONDecodeError:
                    print(f"  [Error] JSON parsing failed on line {line_num + 1} of file {file_name}.")

    # ==========================================
    # 3. Write the aggregated Pool_data.jsonl
    # ==========================================
    output_file_path = os.path.join(output_dir, "Pool_data.jsonl")

    print(f"\n💾 Writing the aggregation results into {output_file_path} ...")

    success_count = 0
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        # Iterate through the aggregation dictionary and write the data for each question to a file
        for data_id, record in merged_data.items():
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            success_count += 1

    print(f"🎉 Mission accomplished! A total of {success_count} Pool records have been successfully integrated.")


if __name__ == "__main__":
    merge_hallucinations_to_pool(FOLDER_PATH, BASE_FILENAME, MODEL_MAPPING)