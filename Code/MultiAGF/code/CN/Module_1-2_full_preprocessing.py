import os
import json

"""
该模块用于对原始数据集进行清洗、过滤、精简字段，并分配唯一连续 ID。

处理步骤：
1. 遍历读取指定文件夹下的所有 jsonl 文件。
2. 过滤掉 answer 和 question 字数不符合要求的数据（answer字数: 140-150, question字数: 70-90）。
3. 移除不需要的字段（options, answer_idx 等）。
4. 为最终留下的数据连续分配 "id" 键，并统一保存到新文件中。
"""

# ==========================================
# 1. 路径与参数配置
# ==========================================
# 输入输出相对目录
input_dir = os.path.join("..", "..", "data", "source_data")
output_dir = os.path.join("..", "..", "data", "intermediate_data", "combined_subset")

# 如果输出目录不存在，则创建该目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 2. 核心处理流程
# ==========================================
# 处理 input_dir 目录下的所有 jsonl 文件
for filename in os.listdir(input_dir):
    if filename.endswith(".jsonl"):
        input_file = os.path.join(input_dir, filename)

        # 将输出文件名设置为以“_cl”结尾
        base_name, extension = os.path.splitext(filename)
        output_filename = f"{base_name}_cl{extension}"
        output_file = os.path.join(output_dir, output_filename)

        print(f"正在处理: {filename} ...")

        # 用于在内存中暂存符合条件的数据
        valid_data_list = []

        # ================= 第一阶段：读取与条件过滤 =================
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # 提取 answer 和 question 字段
                answer = data.get('answer', '')
                question = data.get('question', '')

                # 类型安全检查：确保提取到的内容是字符串
                if not isinstance(answer, str) or not isinstance(question, str):
                    continue

                answer_len = len(answer)
                question_len = len(question)

                # 核心过滤逻辑：answer 在 140-150 之间，且 question 在 70-90 之间
                if not (140 <= answer_len <= 150 and 70 <= question_len <= 90):
                    continue

                # 将无关字段移除
                for key in ['options', 'answer_idx', 'meta_info', 'human_checked', 'human_check_passed']:
                    if key in data:
                        del data[key]

                # 将洗净并符合条件的数据加入候选列表
                valid_data_list.append(data)

        # ================= 第二阶段：分配 ID 并最终保存 =================
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for i, data in enumerate(valid_data_list):
                # 检查是否存在“id”键，如果不存在，则添加一个新的“id”键
                # 在过滤后分配ID，保证 ID 是连续的 (Med01, Med02...)
                if 'id' not in data:
                    new_data = {"id": f"Med{i+1:02d}"}
                    new_data.update(data)
                    data = new_data

                # 写入处理后的数据
                json.dump(data, f_out, ensure_ascii=False)
                f_out.write('\n')

        print(f"  -> 完成: 共成功抽取 {len(valid_data_list)} 条记录，保存至 {output_filename}")

print(f"\n🎉 所有符合条件的 jsonl 文件均已处理完毕，并保存至 {output_dir}")