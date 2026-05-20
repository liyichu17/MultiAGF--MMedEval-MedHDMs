import os
import json
import random
import re

"""
This module is used to clean, filter, reduce the number of fields, and sample the raw dataset, as well as assign unique IDs.

Processing steps:
1. Iterate through and read all JSONL files in the specified folder.
2. Filter out data where the ‘question’ field does not meet the requirements (contains ‘above/below’, contains ‘numbers + dots’, or has fewer than 8 Chinese characters).
3. Remove unnecessary fields (such as `options` and `answer_idx`).
4. Collect the data that meets the criteria; if the total exceeds `sample_size`, perform random sampling.
5. Assign consecutive `id` keys to the final selected data and save it to a new file.
"""

# ==========================================
# 1. Path and parameter configuration
# ==========================================
input_dir = os.path.join("..", "..", "data", "source_data")
output_dir = os.path.join("..", "..", "data", "intermediate_data", "combined_subset")

# Maximum number of samples to be drawn
sample_size = 2000000

# Matches patterns containing numbers followed by a full-width or half-width full stop, such as ‘1.’ and ‘10.’ (supports both half-width full stops \. and full-width full stops ．)
digit_dot_pattern = re.compile(r'\d+[．\.]\s*')

# Chinese Character Statistics: Matching the Commonly Used Chinese Characters Block (CJK Unified Ideographs Basic Block)
CHINESE_RE = re.compile(r'[\u4e00-\u9fff]')


def count_chinese_chars(s: str) -> int:
    """Returns the number of Chinese characters in the string s."""
    if not s:
        return 0
    return len(CHINESE_RE.findall(s))


# If the output directory does not exist, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 2. Core processing workflow
# ==========================================
for filename in os.listdir(input_dir):
    if filename.endswith(".jsonl"):
        input_file = os.path.join(input_dir, filename)

        # Set the output filename to end with “_cl”
        base_name, extension = os.path.splitext(filename)
        output_filename = f"{base_name}_cl{extension}"
        output_file = os.path.join(output_dir, output_filename)

        print(f"Processing: {filename} ...")

        valid_data_list = []

        # ================= Stage 1: Data extraction, cleaning and filtering =================
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                q = data.get('question')

                # --- Filter logic ---
                if isinstance(q, str):
                    # Exclude those containing “above” or “below”
                    if '上' in q or '下' in q:
                        continue
                    # Exclude combinations containing numbers followed by a full stop (e.g. ‘1.’, ‘2.’)
                    if digit_dot_pattern.search(q):
                        continue
                    # The number of Chinese characters must be at least 8
                    if count_chinese_chars(q) < 8:
                        continue
                else:
                    # If question is not a string, skip
                    continue

                # --- Logic for removing unnecessary fields ---
                for key in ['options', 'answer_idx', 'meta_info', 'human_checked', 'human_check_passed']:
                    if key in data:
                        del data[key]

                # Add to shortlist
                valid_data_list.append(data)

        # ================= Stage 2: Random sampling =================
        if len(valid_data_list) > sample_size:
            sampled_data = random.sample(valid_data_list, sample_size)
        else:
            sampled_data = valid_data_list

        # ================= Stage 3: Assigning IDs and final saving =================
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for i, data in enumerate(sampled_data):
                # Check whether an ‘id’ key exists; if not, add a new ‘id’ key
                # Assign IDs after sampling, ensuring that the IDs are consecutive (Med01, Med02...)
                if 'id' not in data:
                    new_data = {"id": f"Med{i + 1:02d}"}
                    new_data.update(data)
                    data = new_data

                # Write the processed data
                json.dump(data, f_out, ensure_ascii=False)
                f_out.write('\n')

        print(
            f"  -> Completed: Originally {len(valid_data_list)} eligible records; {len(sampled_data)} records were ultimately sampled and saved. Saved to {output_filename}")

print(f"\n🎉 All eligible JSONL files have been processed and saved to {output_dir}")