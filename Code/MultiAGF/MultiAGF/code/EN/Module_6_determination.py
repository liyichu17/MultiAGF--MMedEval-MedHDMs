import os
import json
import collections
import pandas as pd
import sys

# ==========================================
# 1. Folder paths and secure output settings
# ==========================================
choicereview_dir = os.path.join("..", "..", "data", "intermediate_data", "consensus_tuple")
rec_dir = os.path.join("..", "..", "data", "intermediate_data", "voting_pool")

# Create a separate output folder to avoid overwriting the original data
output_dir = os.path.join("..", "..", "data", "intermediate_data", "consensus_results")
os.makedirs(output_dir, exist_ok=True)

# Explicitly specifying file name mappings
FILE_MAPPING = {
    "consensus_data.jsonl": "Pool_data.jsonl"
}

model_names = ['Model-A', 'Model-B', 'Model-C', 'Model-D', 'Model-E']

# If a command-line argument is passed, dynamically update the model names
if len(sys.argv) > 1:
    try:
        ext_configs = json.loads(sys.argv[1])
        model_names = [config["model_name"] for config in ext_configs]
        print("💡 The list of model names has been loaded via external parameters.")
    except Exception as e:
        print(f"⚠️ Failed to parse external parameters, using built-in configuration: {e}")

model_keys = ['1', '2', '3', '4', '5']
option_keys = ['A', 'B', 'C', 'D', 'E']


def get_answer_content(rec_item, idx):
    # Revert to your original simplified judgement
    return rec_item.get(idx, '') if idx in option_keys else ''


def all_equal(lst):
    return all(x == lst[0] for x in lst)


# Function to count occurrences (where answer_idx ranges from A to E and answer is the content)
def count_final_answers(data_list, rec_lines_list):
    freq = collections.Counter()
    for item, rec_item in zip(data_list, rec_lines_list):
        idx = item.get('answer_idx', '')
        ans = item.get('answer', '')
        if idx in option_keys and ans:
            choices = [item.get(k, '') for k in model_keys]
            if all_equal(choices):
                for i, k in enumerate(model_keys):
                    if choices[i] == idx:
                        freq[model_names[i]] += 1
            else:
                # Content Group Statistics (Restore to original state)
                contents = [get_answer_content(rec_item, c) for c in choices]
                content_count = collections.Counter(contents)
                max_content_freq = max(content_count.values())
                max_contents = [c for c, v in content_count.items() if v == max_content_freq]

                if len(max_contents) == 1 and max_contents[0] == ans:
                    for i, content in enumerate(contents):
                        if content == ans:
                            freq[model_names[i]] += 1
    return freq


all_data = {}
all_rec = {}
empty_count = 0

print("Processing the file and conducting the initial consensus vote...")

# ==========================================
# 2. Process all JSONL files to determine the initial answer_idx and answer
# ==========================================
for filename in os.listdir(choicereview_dir):
    if not filename.endswith('.jsonl'):
        continue

    choice_path = os.path.join(choicereview_dir, filename)
    rec_filename = FILE_MAPPING.get(filename, filename)
    rec_path = os.path.join(rec_dir, rec_filename)

    if not os.path.exists(rec_path):
        print(f"⚠️ Warning: The corresponding pool file could not be found; skipping: {rec_path}")
        continue

    # Reading data
    with open(choice_path, 'r', encoding='utf-8') as f:
        choice_lines = [json.loads(line) for line in f if line.strip()]
    with open(rec_path, 'r', encoding='utf-8') as f:
        rec_lines = [json.loads(line) for line in f if line.strip()]

    # Process each line
    for i, item in enumerate(choice_lines):
        # Retain only length validation to prevent crashes
        if i >= len(rec_lines):
            item['answer_idx'] = ''
            item['answer'] = ''
            empty_count += 1
            continue

        choices = [item.get(k, '') for k in model_keys]
        rec_item = rec_lines[i]

        if all_equal(choices):
            # Select all the same items, following the original logic
            answer_idx = choices[0]
            item['answer_idx'] = answer_idx
            item['answer'] = get_answer_content(rec_item, answer_idx)
        else:
            # When there are multiple options listed side by side, they should be grouped by content
            contents = [get_answer_content(rec_item, c) for c in choices]
            content_count = collections.Counter(contents)
            max_content_freq = max(content_count.values())
            max_contents = [c for c, v in content_count.items() if v == max_content_freq]

            if len(max_contents) == 1 and max_contents[0]:
                answer_content = max_contents[0]
                answer_idx = None
                for j, content in enumerate(contents):
                    if content == answer_content:
                        answer_idx = choices[j]
                        break
                item['answer_idx'] = answer_idx
                item['answer'] = answer_content
            else:
                item['answer_idx'] = ''
                item['answer'] = ''
                empty_count += 1

    all_data[filename] = choice_lines
    all_rec[filename] = rec_lines

print(f"The number of empty answer_idx entries in the first step：{empty_count}")

# ==========================================
# 3. Calculate the initial frequencies and use the best model to fill in the gaps
# ==========================================
all_items = []
all_rec_items = []
for file, items in all_data.items():
    all_items.extend(items)
    all_rec_items.extend(all_rec[file])

freq1 = count_final_answers(all_items, all_rec_items)
df1 = pd.DataFrame({'模型': list(freq1.keys()), '频次': list(freq1.values())})
df1.to_excel(os.path.join(output_dir, 'model_answer_count1_1.xlsx'), index=False)

freq2 = freq1.copy()
if freq1:
    max_model = freq1.most_common(1)[0][0]
    max_model_idx = model_names.index(max_model)
    print(f"🥇 Using the best-performing model [{max_model}] to fill in the missing answers...")

    for filename, items in all_data.items():
        rec_lines = all_rec[filename]
        for i, item in enumerate(items):
            if not item.get('answer_idx'):
                # Use the selection from the model with the highest current frequency to autocomplete (restore the concise syntax)
                choice = item.get(model_keys[max_model_idx], '')
                content = get_answer_content(rec_lines[i], choice)

                if choice in option_keys and content:
                    item['answer_idx'] = choice
                    item['answer'] = content

                    # Count occurrences as you go: increment by 1 for every model with identical content
                    choices = [item.get(k, '') for k in model_keys]
                    contents = [get_answer_content(rec_lines[i], c) for c in choices]
                    for idx, c_content in enumerate(contents):
                        if c_content == content and choices[idx] in option_keys:
                            freq2[model_names[idx]] += 1
                else:
                    item['answer_idx'] = ''
                    item['answer'] = ''

        # Save all output files to a secure directory, naming them “original_name_final.jsonl”
        final_filename = filename.replace(".jsonl", "_final.jsonl")
        with open(os.path.join(output_dir, final_filename), 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

# ==========================================
# 4. Output the frequency statistics after final completion
# ==========================================
model_names_out = list(freq2.keys())
model_counts_out = list(freq2.values())
df2 = pd.DataFrame({'模型': model_names_out, '频次': model_counts_out})
df2.to_excel(os.path.join(output_dir, 'model_answer_count2_1.xlsx'), index=False)

print(f'✅ Processing complete. The data and statistical results have been securely saved to：{output_dir}')