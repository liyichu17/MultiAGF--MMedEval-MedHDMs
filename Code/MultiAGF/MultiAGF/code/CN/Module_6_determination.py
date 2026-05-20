import os
import json
import collections
import pandas as pd
import sys

# ==========================================
# 1. 文件夹路径与安全输出设置
# ==========================================
choicereview_dir = os.path.join("..", "..", "data", "intermediate_data", "consensus_tuple")
rec_dir = os.path.join("..", "..", "data", "intermediate_data", "voting_pool")

# 创建独立的输出文件夹，避免覆盖原始数据
output_dir = os.path.join("..", "..", "data", "intermediate_data", "consensus_results")
os.makedirs(output_dir, exist_ok=True)

# 显式指定文件名映射
FILE_MAPPING = {
    "consensus_data.jsonl": "Pool_data.jsonl"
}

model_names = ['Model-A', 'Model-B', 'Model-C', 'Model-D', 'Model-E']

# 如果有命令行参数传递，则动态更新模型名称
if len(sys.argv) > 1:
    try:
        ext_configs = json.loads(sys.argv[1])
        model_names = [config["model_name"] for config in ext_configs]
        print("💡 已通过外部参数加载模型名称列表。")
    except Exception as e:
        print(f"⚠️ 外部参数解析失败，使用内置配置: {e}")

model_keys = ['1', '2', '3', '4', '5']
option_keys = ['A', 'B', 'C', 'D', 'E']


def get_answer_content(rec_item, idx):
    # 恢复为您最初的精简判断
    return rec_item.get(idx, '') if idx in option_keys else ''


def all_equal(lst):
    return all(x == lst[0] for x in lst)


# 统计频次的函数（answer_idx为A~E，answer为内容）
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
                # 内容分组统计（恢复原样）
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

print("正在处理文件，进行初始共识投票...")

# ==========================================
# 2. 处理所有jsonl文件，确定初始 answer_idx 和 answer
# ==========================================
for filename in os.listdir(choicereview_dir):
    if not filename.endswith('.jsonl'):
        continue

    choice_path = os.path.join(choicereview_dir, filename)
    rec_filename = FILE_MAPPING.get(filename, filename)
    rec_path = os.path.join(rec_dir, rec_filename)

    if not os.path.exists(rec_path):
        print(f"⚠️ 警告: 找不到对应的 pool 文件，跳过: {rec_path}")
        continue

    # 读取数据
    with open(choice_path, 'r', encoding='utf-8') as f:
        choice_lines = [json.loads(line) for line in f if line.strip()]
    with open(rec_path, 'r', encoding='utf-8') as f:
        rec_lines = [json.loads(line) for line in f if line.strip()]

    # 处理每一行
    for i, item in enumerate(choice_lines):
        # 仅保留长度校验，防止崩溃
        if i >= len(rec_lines):
            item['answer_idx'] = ''
            item['answer'] = ''
            empty_count += 1
            continue

        choices = [item.get(k, '') for k in model_keys]
        rec_item = rec_lines[i]

        if all_equal(choices):
            # 全部选择一样，按原逻辑
            answer_idx = choices[0]
            item['answer_idx'] = answer_idx
            item['answer'] = get_answer_content(rec_item, answer_idx)
        else:
            # 多个选项并列最多，需按内容分组
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

print(f"第一步置空的 answer_idx 数量：{empty_count}")

# ==========================================
# 3. 统计初始频次，并使用最佳模型进行兜底补全
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
    print(f"🥇 正在使用表现最佳的模型 [{max_model}] 补全空缺答案...")

    for filename, items in all_data.items():
        rec_lines = all_rec[filename]
        for i, item in enumerate(items):
            if not item.get('answer_idx'):
                # 直接用当前频次最高模型的选择补全（恢复精简写法）
                choice = item.get(model_keys[max_model_idx], '')
                content = get_answer_content(rec_lines[i], choice)

                if choice in option_keys and content:
                    item['answer_idx'] = choice
                    item['answer'] = content

                    # 边补全边统计频次：所有内容一致的模型都+1
                    choices = [item.get(k, '') for k in model_keys]
                    contents = [get_answer_content(rec_lines[i], c) for c in choices]
                    for idx, c_content in enumerate(contents):
                        if c_content == content and choices[idx] in option_keys:
                            freq2[model_names[idx]] += 1
                else:
                    item['answer_idx'] = ''
                    item['answer'] = ''

        # 统一输出文件到安全目录，文件名为 原名_final.jsonl
        final_filename = filename.replace(".jsonl", "_final.jsonl")
        with open(os.path.join(output_dir, final_filename), 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

# ==========================================
# 4. 输出最终补全后频次统计
# ==========================================
model_names_out = list(freq2.keys())
model_counts_out = list(freq2.values())
df2 = pd.DataFrame({'模型': model_names_out, '频次': model_counts_out})
df2.to_excel(os.path.join(output_dir, 'model_answer_count2_1.xlsx'), index=False)

print(f'✅ 处理完成，数据与统计结果已安全保存至：{output_dir}')