import os
import json
import sys

"""
总结：用于根据投票决定的选项来拼出最终的幻觉数据。

该代码会遍历 consensus_results 文件夹（即投票结果文件夹）中的 jsonl 文件。
读取每行数据的 answer_idx 字段（A、B、C、D、E），
然后去 hallucinated_answers 文件夹中，找到对应模型生成的幻觉文件，提取同一行的数据。
最终输出的文件会合并保存在 final_data 文件夹中。
"""

# ==========================================
# 1. 文件夹与映射配置
# ==========================================
consensus_dir = os.path.join("..", "..", "data", "intermediate_data", "consensus_results")
hallucinated_dir = os.path.join("..", "..", "data", "intermediate_data", "hallucinated_answers")
output_dir = os.path.join("..", "..", "data", "final_data")

os.makedirs(output_dir, exist_ok=True)

# 选项 A~E 与模型名称的映射关系
# 请确保这里的名称与生成幻觉数据时的模型名称完全一致
MODEL_MAP = {
    "A": "Model-A",
    "B": "Model-B",
    "C": "Model-C",
    "D": "Model-D",
    "E": "Model-E"
}

# 如果有命令行参数传递，则动态更新模型映射
if len(sys.argv) > 1:
    try:
        ext_configs = json.loads(sys.argv[1])
        new_map = {}
        keys = ["A", "B", "C", "D", "E"]
        for i, config in enumerate(ext_configs):
            if i < len(keys):
                new_map[keys[i]] = config["model_name"]
        MODEL_MAP = new_map
        print("💡 已通过外部参数动态加载模型名称映射。")
    except Exception as e:
        print(f"⚠️ 外部参数解析失败，使用内置配置: {e}")

# 幻觉文件的基础名称（统一后缀）
BASE_HALLUCINATED_NAME = "MedBench3000_cl_hallucinated.jsonl"

# 如果有第二个命令行参数传递，则覆盖幻觉文件名配置
if len(sys.argv) > 2:
    try:
        base_configs = json.loads(sys.argv[2])
        BASE_HALLUCINATED_NAME = base_configs.get("base_filename", BASE_HALLUCINATED_NAME)
        print(f"💡 已通过外部参数加载基础文件名: {BASE_HALLUCINATED_NAME}")
    except Exception as e:
        print(f"⚠️ 基础文件名参数解析失败: {e}")

def get_jsonl_files(folder):
    return [f for f in os.listdir(folder) if f.endswith('.jsonl')]


# ==========================================
# 2. 主处理流程
# ==========================================
print("正在根据投票结果拼装最终的幻觉数据集...\n")

for filename in get_jsonl_files(consensus_dir):
    choicereview_path = os.path.join(consensus_dir, filename)

    # ==========================================
    # 2.1 预读取 5 个模型生成的幻觉文件内容到内存中
    # ==========================================
    model_lines = {}
    for option_key, model_name in MODEL_MAP.items():
        # 兼容模型名称中的特殊字符（与生成文件时的安全命名逻辑一致）
        safe_model_name = model_name.replace('/', '_').replace(':', '_')

        # 拼接目标文件路径：例如 "qwen3.6-flash_MedBench10_cl_hallucinated.jsonl"
        model_filename = f"{safe_model_name}_{BASE_HALLUCINATED_NAME}"
        model_path = os.path.join(hallucinated_dir, model_filename)

        if os.path.exists(model_path):
            with open(model_path, 'r', encoding='utf-8') as f:
                model_lines[option_key] = f.readlines()
        else:
            print(f"⚠️ 警告: 找不到模型 [{option_key}] 的幻觉文件: {model_path}")
            model_lines[option_key] = []

    # ==========================================
    # 2.2 处理投票结果文件，逐行组装最终数据
    # ==========================================
    with open(choicereview_path, 'r', encoding='utf-8') as f:
        review_lines = f.readlines()

    output_lines = []
    for idx, line in enumerate(review_lines):
        try:
            data = json.loads(line)
        except Exception:
            output_lines.append('{}')
            continue

        answer_idx = data.get('answer_idx', None)

        # 判断：如果该行有合法的投票结果 (A~E)，且对应模型的幻觉文件数据够长
        if answer_idx in MODEL_MAP and idx < len(model_lines[answer_idx]):
            # 提取被选中模型生成的对应行幻觉数据，并去除末尾换行符
            output_lines.append(model_lines[answer_idx][idx].rstrip('\n'))
        else:
            # 如果没有统一意见被置空，或者该行数据缺失，输出空对象
            output_lines.append('{}')

    # ==========================================
    # 2.3 输出组装完毕的数据
    # ==========================================
    # 如果读取的是 _final.jsonl，输出时可以去掉这个后缀让文件名更干净
    # clean_filename = filename.replace("_final.jsonl", ".jsonl")
    clean_filename = "final_hal_data.jsonl"
    output_path = os.path.join(output_dir, clean_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        for l in output_lines:
            f.write(l + '\n')

    print(f"✅ 文件 {clean_filename} 处理完成。")

print(f"\n🎉 所有拼装任务完成！最终数据已输出至目录：{output_dir}")