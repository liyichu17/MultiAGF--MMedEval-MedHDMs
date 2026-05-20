import os
import json
import sys

# ==========================================
# 1. 配置参数与文件映射逻辑
# ==========================================
FOLDER_PATH = os.path.join("..", "..", "data", "intermediate_data", "hallucinated_answers")
output_dir = os.path.join("..", "..", "data", "intermediate_data", "voting_pool")

# 这里填写你刚才批量生成时，原始文件的基础名称。
# 比如原始文件是 "med_data.jsonl"，经过上一步后各模型的后缀变成了 "_hallucinated.jsonl"
# 所以寻找的基础文件名应该是 "med_data_hallucinated.jsonl"
BASE_FILENAME = "MedBench3000_cl_hallucinated.jsonl"  # 请替换为您实际的基础文件名

# 如果有第二个命令行参数传递，则覆盖基础文件名配置
if len(sys.argv) > 2:
    try:
        base_configs = json.loads(sys.argv[2])
        BASE_FILENAME = base_configs.get("base_filename", BASE_FILENAME)
        print(f"💡 已通过外部参数加载基础文件名: {BASE_FILENAME}")
    except Exception as e:
        print(f"⚠️ 基础文件名参数解析失败: {e}")

# 核心逻辑：定义 A~E 与 5 个模型的映射关系
MODEL_MAPPING = {
    "A": "Model-A",
    "B": "Model-B",
    "C": "Model-C",
    "D": "Model-D",
    "E": "Model-E"
}

# 如果有命令行参数传递，则动态更新模型名称
if len(sys.argv) > 1:
    try:
        ext_configs = json.loads(sys.argv[1])
        # 假设 ext_configs 是一个列表，按顺序对应 A, B, C, D, E
        # 或者根据列表中 model_name 的 key 顺序进行映射
        new_mapping = {}
        keys = ["A", "B", "C", "D", "E"]
        for i, config in enumerate(ext_configs):
            if i < len(keys):
                new_mapping[keys[i]] = config["model_name"]
        MODEL_MAPPING = new_mapping
        print("💡 已通过外部参数动态加载模型映射。")
    except Exception as e:
        print(f"⚠️ 外部参数解析失败，使用内置配置: {e}")


# ==========================================
# 2. 核心提取与合并逻辑
# ==========================================
def merge_hallucinations_to_pool(folder_path, base_filename, mapping):
    # 用于存储聚合后数据的全局字典，主键为 id
    merged_data = {}

    print(f"正在扫描 {folder_path}，准备提取合并 Pool_data...\n")

    # 遍历字典中的每一个选项 (A, B, C, D, E) 及其对应的模型名称
    for option_key, model_name in mapping.items():
        # 构造对应的文件名 (与上一步的 safe_model_name 生成逻辑保持一致)
        safe_model_name = model_name.replace('/', '_').replace(':', '_')
        file_name = f"{safe_model_name}_{base_filename}"
        file_path = os.path.join(folder_path, file_name)

        if not os.path.exists(file_path):
            print(f"⚠️ 警告: 找不到选项 [{option_key}] 对应的文件: {file_name}")
            continue

        print(f"📖 正在提取 [{option_key}] 选项数据，来源模型: {model_name} ...")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    data_id = data.get("id", "")
                    question = data.get("question", "")
                    hallucinated_ans = data.get("Hallucinated_answer", "")

                    if not data_id:
                        print(f"  [警告] 发现缺少 id 的数据，行号: {line_num + 1}，已跳过。")
                        continue

                    # 如果这个 id 是第一次出现，初始化它的基础字典结构
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

                    # 将提取到的 Hallucinated_answer 填入对应的 A/B/C/D/E 键中
                    merged_data[data_id][option_key] = hallucinated_ans

                except json.JSONDecodeError:
                    print(f"  [错误] 文件 {file_name} 第 {line_num + 1} 行 JSON 解析失败。")

    # ==========================================
    # 3. 写入聚合后的 Pool_data.jsonl
    # ==========================================
    output_file_path = os.path.join(output_dir, "Pool_data.jsonl")

    print(f"\n💾 正在将聚合结果写入 {output_file_path} ...")

    success_count = 0
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        # 遍历聚合字典，将每一道题的数据写入文件
        for data_id, record in merged_data.items():
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            success_count += 1

    print(f"🎉 任务完成！共成功整合 {success_count} 条 Pool 数据。")


if __name__ == "__main__":
    merge_hallucinations_to_pool(FOLDER_PATH, BASE_FILENAME, MODEL_MAPPING)