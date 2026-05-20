import json
import os

"""
该模块用于从每个数据集的所有数据记录中移除不需要的字段，
同时为每一行数据添加一个“id”键。
"""
# 输入输出相对目录
input_dir = os.path.join("..", "..", "data", "source_data")
output_dir = os.path.join("..", "..", "data", "intermediate_data", "combined_subset")

# 如果输出目录不存在，则创建该目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 处理 input_dir 目录下的所有 jsonl 文件
for filename in os.listdir(input_dir):
    if filename.endswith(".jsonl"):
        input_file = os.path.join(input_dir, filename)

        # 将输出文件名设置为以“_cl”结尾
        base_name, extension = os.path.splitext(filename)
        output_filename = f"{base_name}_cl{extension}"
        output_file = os.path.join(output_dir, output_filename)

        print(f"Processing: {filename} -> {output_filename}")

        # 读取和处理数据
        with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
            for i, line in enumerate(f_in):
                if line.strip():
                    data = json.loads(line)

                    # 将无关字段置于“[]”中以将其移除
                    for key in ['options', 'answer_idx', 'meta_info', 'human_checked', 'human_check_passed']:
                        if key in data:
                            del data[key]

                    # 检查是否存在“id”键，如果不存在，则添加一个新的“id”键
                    if 'id' not in data:
                        # 生成一个新的排序字典，以确保 ID 位于开头
                        # ID 格式为“MedXX”，其中 XX 是从 01 开始的、首位补零的数字（第一条记录）
                        new_data = {"id": f"Med{i+1:02d}"}
                        new_data.update(data)
                        data = new_data

                    # 写入处理后的数据
                    json.dump(data, f_out, ensure_ascii=False)
                    f_out.write('\n')

print(f"所有符合条件的 jsonl 文件均已处理完毕，并保存至 {output_dir}")
