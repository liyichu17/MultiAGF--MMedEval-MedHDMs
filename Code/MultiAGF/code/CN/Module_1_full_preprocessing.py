import os
import json
import random
import re

"""
该模块用于对原始数据集进行清洗、过滤、精简字段、抽样，并分配唯一 ID。

处理步骤：
1. 遍历读取指定文件夹下的所有 jsonl 文件。
2. 过滤掉 question 不符合要求的数据（包含"上/下"、包含"数字+点"、汉字少于8个）。
3. 移除不需要的字段（options, answer_idx 等）。
4. 将符合条件的数据收集起来，若超出 sample_size 则进行随机抽样。
5. 为最终选定的数据连续分配 "id" 键，并保存到新文件中。
"""

# ==========================================
# 1. 路径与参数配置
# ==========================================
input_dir = os.path.join("..", "..", "data", "source_data")
output_dir = os.path.join("..", "..", "data", "intermediate_data", "combined_subset")

# 要抽取的最大样本数量
sample_size = 2000000

# 匹配数字加点的模式，例如 "1." "10." 等（兼容了半角点 \. 和全角点 ．）
digit_dot_pattern = re.compile(r'\d+[．\.]\s*')

# 汉字统计：匹配常用汉字区（CJK 统一汉字基本区）
CHINESE_RE = re.compile(r'[\u4e00-\u9fff]')


def count_chinese_chars(s: str) -> int:
    """返回字符串 s 中汉字字符的数量。"""
    if not s:
        return 0
    return len(CHINESE_RE.findall(s))


# 如果输出目录不存在，则创建该目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 2. 核心处理流程
# ==========================================
for filename in os.listdir(input_dir):
    if filename.endswith(".jsonl"):
        input_file = os.path.join(input_dir, filename)

        # 将输出文件名设置为以“_cl”结尾
        base_name, extension = os.path.splitext(filename)
        output_filename = f"{base_name}_cl{extension}"
        output_file = os.path.join(output_dir, output_filename)

        print(f"正在处理: {filename} ...")

        valid_data_list = []

        # ================= 第一阶段：读取与清洗过滤 =================
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

                # --- 过滤逻辑 ---
                if isinstance(q, str):
                    # 排除含有 '上' 或 '下'
                    if '上' in q or '下' in q:
                        continue
                    # 排除含有数字加点的组合（如 "1.", "2."）
                    if digit_dot_pattern.search(q):
                        continue
                    # 汉字数量必须 >= 8
                    if count_chinese_chars(q) < 8:
                        continue
                else:
                    # 如果 question 不是字符串，跳过
                    continue

                # --- 删除无用字段逻辑 ---
                for key in ['options', 'answer_idx', 'meta_info', 'human_checked', 'human_check_passed']:
                    if key in data:
                        del data[key]

                # 加入候选列表
                valid_data_list.append(data)

        # ================= 第二阶段：随机抽样 =================
        if len(valid_data_list) > sample_size:
            sampled_data = random.sample(valid_data_list, sample_size)
        else:
            sampled_data = valid_data_list

        # ================= 第三阶段：分配 ID 并最终保存 =================
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for i, data in enumerate(sampled_data):
                # 检查是否存在“id”键，如果不存在，则添加一个新的“id”键
                # 在抽样后分配ID，保证 ID 是连续的 (Med01, Med02...)
                if 'id' not in data:
                    new_data = {"id": f"Med{i + 1:02d}"}
                    new_data.update(data)
                    data = new_data

                # 写入处理后的数据
                json.dump(data, f_out, ensure_ascii=False)
                f_out.write('\n')

        print(
            f"  -> 完成: 原始符合条件 {len(valid_data_list)} 条，最终抽取保存 {len(sampled_data)} 条。保存至 {output_filename}")

print(f"\n🎉 所有符合条件的 jsonl 文件均已处理完毕，并保存至 {output_dir}")