import os
import json
import re
import sys
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. 配置 5 个大模型的独立参数 (API, URL, 模型名)
# ==========================================
# 默认配置，如果没有通过参数传递则使用此配置
MODELS_CONFIG = [
    {
        "model_name": "Model-A",
        "api_key": "YOUR_API_KEY_1",
        "base_url": "https://api.model1.com/v1"
    },
    {
        "model_name": "Model-B",
        "api_key": "YOUR_API_KEY_2",
        "base_url": "https://api.model2.com/v1"
    },
    {
        "model_name": "Model-C",
        "api_key": "YOUR_API_KEY_3",
        "base_url": "https://api.model3.com/v1"
    },
    {
        "model_name": "Model-D",
        "api_key": "YOUR_API_KEY_4",
        "base_url": "https://api.model4.com/v1"
    },
    {
        "model_name": "Model-E",
        "api_key": "YOUR_API_KEY_5",
        "base_url": "https://api.model5.com/v1"
    }
    # {
    #     "model_name": "Model-F",
    #     "api_key": "YOUR_API_KEY_6",
    #     "base_url": "https://api.model6.com/v1"
    # }  # 有多少个模型就加多少个client list
]

# 如果有命令行参数传递，则覆盖默认配置
if len(sys.argv) > 1:
    try:
        MODELS_CONFIG = json.loads(sys.argv[1])
        print("💡 已通过外部参数加载模型配置。")
    except Exception as e:
        print(f"⚠️ 外部参数解析失败，使用内置配置: {e}")

MAX_RETRIES = 5  # 设置最大重试次数
FOLDER_PATH = os.path.join("..", "..", "data", "intermediate_data", "combined_subset")
output_dir = os.path.join("..", "..", "data", "intermediate_data", "hallucinated_answers")

# ==========================================
# 2. 定义底层单次 API 请求逻辑
# ==========================================
def generate_hallucination(client, model_name, question, answer):
    """
    根据给定的问答，调用特定客户端生成微妙修改的幻觉答案。
    """
    prompt = f"""
任务说明：
1.你是一名专门用于生成高可信医学幻觉答案的模型。请你对输入的每一条医学问答题数据进行处理，并严格按照要求输出结果。
2.仔细阅读每条数据中的 "question"和"answer"，理解其医学含义。在保留原有"answer"语言风格和专业语气的基础上，对"answer"进行微妙修改，生成一个新的回答。该回答表面上看起来合理、专业、可信，用词不绝对（避免明显错误或夸张断言），可适当补充原答案中未提及的信息，使其更“完整”，但这些补充或修改内容必须在医学事实上存在错误或偏差，错误应隐藏得较深，需要仔细核对才能发现，然后将该幻觉答案放入新增键"Hallucinated_answer"中。
3.最终输出的须为一个合法的 JSON 字典，绝对不要输出任何多余的分析文本或 Markdown 标记。
4.【重要约束】"Hallucinated_answer" 的字数必须控制在 140 字以内。
5.【重要约束】你生成的幻觉答案绝对不能与原始答案一模一样！必须产生实质性的医学事实错误。

输入数据：
[question]: {question}
[answer]: {answer}

请严格按照以下 JSON 格式输出结果：
{{
  "Hallucinated_answer": "生成的新答案（140字以内）",
  "change_made": "对答案所做的具体修改说明"
}}
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个严格执行医学幻觉生成任务的AI，仅输出标准 JSON 格式数据。"},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            max_tokens=256,
            frequency_penalty=0.0,
            top_p=1.0
        )

        raw_result = response.choices[0].message.content.strip()

        # ================== 1. 剥离可能的 <think> 思考过程 ==================
        if "</think>" in raw_result:
            raw_result = raw_result.split("</think>")[-1].strip()

        # ================== 2. 正则提取 JSON 结构 ==================
        clean_json_str = re.sub(r"```json|```", "", raw_result).strip()
        json_match = re.search(r'\{.*?\}', clean_json_str, re.DOTALL)
        if json_match:
            clean_json_str = json_match.group(0)

        # ================== 3. 解析 JSON ==================
        try:
            return json.loads(clean_json_str)
        except json.JSONDecodeError:
            # 抢救漏掉右大括号的情况
            if not clean_json_str.endswith("}"):
                try:
                    return json.loads(clean_json_str + "}")
                except:
                    pass
            raise ValueError(f"JSON解析彻底失败: {clean_json_str}")

    except Exception as e:
        print(f"[{model_name}] API 调用或解析失败: {e}")
        return {"Hallucinated_answer": "生成失败", "change_made": "生成失败"}


# ==========================================
# 3. 定义单个模型的完整文件处理工作流
# ==========================================
def process_single_model(config, folder_path, out_dir):
    """
    此函数将在独立线程中运行，负责某个特定模型的所有数据处理。
    """
    model_name = config["model_name"]
    api_key = config["api_key"]
    base_url = config["base_url"]

    # 为当前模型实例化专属的 OpenAI Client
    client = OpenAI(api_key=api_key, base_url=base_url)

    safe_model_name = model_name.replace('/', '_').replace(':', '_')

    print(f"🚀 [线程启动] 模型: {model_name} 已就绪，开始扫描文件...")

    try:
        for filename in os.listdir(folder_path):
            # 仅处理 jsonl 文件，并且跳过已经处理过的文件，避免重复套娃
            if filename.endswith(".jsonl") and "_hallucinated" not in filename:
                input_file_path = os.path.join(folder_path, filename)
                output_filename = f"{safe_model_name}_{filename.replace('.jsonl', '_hallucinated.jsonl')}"
                output_file_path = os.path.join(out_dir, output_filename)

                if os.path.exists(output_file_path):
                    print(f"⚠️ [{model_name}] 文件 {output_filename} 已存在，跳过。")
                    continue

                with open(input_file_path, 'r', encoding='utf-8') as infile, \
                        open(output_file_path, 'w', encoding='utf-8') as outfile:

                    for idx, line in enumerate(infile):
                        if not line.strip(): continue
                        data = json.loads(line)

                        data_id = data.get("id", "")
                        question = data.get("question", "")
                        answer = data.get("answer", "")

                        print(f"  [{model_name}] 正在处理: {filename} -> 第 {idx + 1} 条 (ID: {data_id})")

                        llm_result = None

                        # 引入重试机制
                        for attempt in range(MAX_RETRIES):
                            # 调用 API 获取幻觉数据 (移除了 rationale)
                            llm_result = generate_hallucination(client, model_name, question, answer)

                            # 检查 1：返回格式是否错误，或者内部表明生成失败
                            if not isinstance(llm_result, dict) or llm_result.get("Hallucinated_answer") in ["生成失败", "生成异常"]:
                                print(f"    [{model_name}] 生成异常，重试 {attempt + 1}/{MAX_RETRIES}...")
                                continue

                            hallucinated_ans = llm_result.get("Hallucinated_answer", "")

                            # 检查 2：生成的幻觉答案是否与原答案完全一样
                            if hallucinated_ans.strip() == answer.strip():
                                print(f"    [{model_name}] 答案未实质性修改，重试 {attempt + 1}/{MAX_RETRIES}...")
                                continue

                            # 检查 3：字数检查提示
                            if len(hallucinated_ans) > 140:
                                print(f"    [{model_name}] 提示: 幻觉答案字数({len(hallucinated_ans)})超过140字，但仍予以保留。")

                            # 如果上述检查都通过了，说明生成成功，跳出重试循环
                            break
                        else:
                            print(f"    [{model_name}] ❌ 达到最大重试次数，ID: {data_id} 生成质量不佳。")
                            if not isinstance(llm_result, dict):
                                llm_result = {"Hallucinated_answer": "生成异常", "change_made": "多次重试后依旧异常"}

                        # 按照要求组装最终输出格式
                        output_data = {
                            "id": data_id,
                            "question": question,
                            "answer": answer,
                            "Hallucinated_answer": llm_result.get("Hallucinated_answer", ""),
                            "change_made": llm_result.get("change_made", "")
                        }

                        # 写入新的 jsonl 文件
                        outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')

                print(f"✅ [{model_name}] 文件 {filename} 处理完成！")

    except Exception as e:
        return f"❌ [{model_name}] 线程发生致命错误: {e}"

    return f"🎉 [{model_name}] 所有任务圆满完成！"


# ==========================================
# 4. 主程序：启动多线程并行执行
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(FOLDER_PATH):
        print(f"错误：找不到文件夹 '{FOLDER_PATH}'。")
    else:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        print(f"准备在 {FOLDER_PATH} 下并发启动 {len(MODELS_CONFIG)} 个模型任务...\n")

        # 使用 ThreadPoolExecutor 实现并行，最大线程数设定为模型的数量
        with ThreadPoolExecutor(max_workers=len(MODELS_CONFIG)) as executor:
            # 提交所有的模型任务到线程池
            futures = [executor.submit(process_single_model, config, FOLDER_PATH, output_dir) for config in MODELS_CONFIG]

            # 等待并捕获每个线程的最终返回结果
            for future in as_completed(futures):
                result_message = future.result()
                print(f"\n{result_message}")

        print("\n🏆 所有并发模型生成任务已全部结束！")