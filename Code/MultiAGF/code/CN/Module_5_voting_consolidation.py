import os
import json
import sys

# ==========================================
# 1. 配置参数与文件映射逻辑
# ==========================================
FOLDER_PATH = os.path.join("..", "..", "data", "intermediate_data", "voting_results")
output_dir = os.path.join("..", "..", "data", "intermediate_data", "consensus_tuple")

# 这里填写你刚才批量进行作答投票时，原始文件的基础名称。
# 例如，如果处理的是 voted 文件，这里的基础文件名可能是 "MedBench10_cl_voted.jsonl"
BASE_FILENAME = "Pool_data_voted.jsonl"  # 请替换为您实际的基础文件名

# 核心逻辑：定义 1~5 与 5 个模型的映射关系
MODEL_MAPPING = {
    "1": "Model-A",
    "2": "Model-B",
    "3": "Model-C",
    "4": "Model-D",
    "5": "Model-E"
}

# 如果有命令行参数传递，则动态更新模型名称
if len(sys.argv) > 1:
    try:
        ext_configs = json.loads(sys.argv[1])
        new_mapping = {}
        # 对应键位 1, 2, 3, 4, 5
        keys = ["1", "2", "3", "4", "5"]
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
def merge_votes_to_result(folder_path, base_filename, mapping):
    # 用于存储聚合后数据的全局字典，主键为 id
    merged_data = {}

    print(f"正在扫描 {folder_path}，准备提取合并投票结果至 result_data...\n")

    # 遍历字典中的每一个选项 (1, 2, 3, 4, 5) 及其对应的模型名称
    for option_key, model_name in mapping.items():
        # 构造对应的文件名 (与之前 safe_model_name 生成逻辑保持一致)
        safe_model_name = model_name.replace('/', '_').replace(':', '_')
        file_name = f"{safe_model_name}_{base_filename}"
        file_path = os.path.join(folder_path, file_name)

        if not os.path.exists(file_path):
            print(f"⚠️ 警告: 找不到键位 [{option_key}] 对应的文件: {file_name}")
            continue

        print(f"📖 正在提取键位 [{option_key}] 的投票数据，来源模型: {model_name} ...")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    data_id = data.get("id", "")
                    question = data.get("question", "")

                    # 修改：这里提取的是模型的投票答案 (answer)
                    vote_ans = data.get("answer", "")

                    if not data_id:
                        print(f"  [警告] 发现缺少 id 的数据，行号: {line_num + 1}，已跳过。")
                        continue

                    # 如果这个 id 是第一次出现，初始化它的基础字典结构
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

                    # 将提取到的 answer 填入对应的 1/2/3/4/5 键中
                    merged_data[data_id][option_key] = vote_ans

                except json.JSONDecodeError:
                    print(f"  [错误] 文件 {file_name} 第 {line_num + 1} 行 JSON 解析失败。")

    # ==========================================
    # 3. 写入聚合后的 consensus_data.jsonl
    # ==========================================
    output_file_path = os.path.join(output_dir, "consensus_data.jsonl")

    print(f"\n💾 正在将聚合结果写入 {output_file_path} ...")

    success_count = 0
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        # 遍历聚合字典，将每一道题的数据写入文件
        for data_id, record in merged_data.items():
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            success_count += 1

    print(f"🎉 任务完成！共成功整合 {success_count} 条投票结果数据。")


if __name__ == "__main__":
    merge_votes_to_result(FOLDER_PATH, BASE_FILENAME, MODEL_MAPPING)