import os
import json

"""
This module is used to clean, filter and streamline fields in the raw dataset, and to assign unique, sequential IDs.

Processing steps:
1. Iterate through and read all JSONL files in the specified folder.
2. Filter out data where the character counts for ‘answer’ and ‘question’ do not meet the requirements (answer character count: 140–150; question character count: 70–90).
3. Remove unnecessary fields (such as `options` and `answer_idx`).
4. Assign a sequential ‘id’ key to the remaining data and save it to a new file.
"""

# ==========================================
# 1. Path and parameter configuration
# ==========================================
# Relative input and output directories
input_dir = os.path.join("..", "..", "data", "source_data")
output_dir = os.path.join("..", "..", "data", "intermediate_data", "combined_subset")

# If the output directory does not exist, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 2. Core processing workflow
# ==========================================
# Process all JSONL files in the input_dir directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jsonl"):
        input_file = os.path.join(input_dir, filename)

        # Set the output filename to end with “_cl”
        base_name, extension = os.path.splitext(filename)
        output_filename = f"{base_name}_cl{extension}"
        output_file = os.path.join(output_dir, output_filename)

        print(f"Processing: {filename} ...")

        # 用于在内存中暂存符合条件的数据
        valid_data_list = []

        # ================= Stage 1: Reading and conditional filtering =================
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract the “answer” and “question” fields
                answer = data.get('answer', '')
                question = data.get('question', '')

                # Type safety check: Ensure that the extracted content is a string
                if not isinstance(answer, str) or not isinstance(question, str):
                    continue

                answer_len = len(answer)
                question_len = len(question)

                # Core filtering logic: where “answer” is between 140 and 150, and “question” is between 70 and 90
                if not (140 <= answer_len <= 150 and 70 <= question_len <= 90):
                    continue

                # Remove irrelevant fields
                for key in ['options', 'answer_idx', 'meta_info', 'human_checked', 'human_check_passed']:
                    if key in data:
                        del data[key]

                # Add the cleaned and eligible data to the candidate list
                valid_data_list.append(data)

        # ================= Stage 2: Assigning IDs and final saving =================
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for i, data in enumerate(valid_data_list):
                # Check whether an “id” key exists; if not, add a new “id” key
                # Assign an ID after filtering, ensuring that the IDs are consecutive (Med01, Med02...)
                if 'id' not in data:
                    new_data = {"id": f"Med{i+1:02d}"}
                    new_data.update(data)
                    data = new_data

                # Write the processed data
                json.dump(data, f_out, ensure_ascii=False)
                f_out.write('\n')

        print(f"  -> Completed: A total of {len(valid_data_list)} records were successfully extracted and saved to {output_filename}")

print(f"\n🎉 All eligible JSONL files have been processed and saved to {output_dir}")