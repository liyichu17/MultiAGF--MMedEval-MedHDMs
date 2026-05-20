import os
import json
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

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

MAX_RETRIES = 5  # 设置最大重试次数，推荐稍微设高点，保障输出格式稳定
FOLDER_PATH = os.path.join("..", "..", "data", "intermediate_data", "voting_pool")
output_dir = os.path.join("..", "..", "data", "intermediate_data", "voting_results")

# ==========================================
# 2. 定义底层单次 API 请求逻辑 (作答与投票)
# ==========================================
def answer_medical_question(client, model_name, question, opt_a, opt_b, opt_c, opt_d, opt_e):
    """
    根据给定的医学题目及五个选项，让大模型推理并选出正确答案及理由。
    """
    prompt = f"""
任务说明：
1.你是一名医学问答领域专家，精通医学基础知识、临床医学及相关学科内容。请你对输入的每一条医学选择题数据进行分析，并严格按照要求输出结果。
2.仔细阅读每条数据中的 "question"。结合医学专业知识，对选项 "A"、"B"、"C"、"D"、"E" 进行分析。从五个选项中选择唯一一个最恰当的答案，然后将你的选择放入新增键"answer"中，并将选择该选项的理由放入新增键"reason"中。
3.最终输出的须为合法的 JSON 格式，包含 "answer" 和 "reason" 两个键。
"answer"：仅填写所选选项的大写字母（A、B、C、D 或 E）。
"reason"：用简洁、专业、逻辑清晰的语言说明选择该选项的医学依据，不需要做出勘误。
绝对不要输出任何多余的分析文本或 Markdown 标记。

输入数据：
[question]: {question}
[A]: {opt_a}
[B]: {opt_b}
[C]: {opt_c}
[D]: {opt_d}
[E]: {opt_e}

请严格按照以下 JSON 格式输出结果：
{{
  "answer": "你的选择(A/B/C/D/E)",
  "reason": "你的医学依据说明"
}}
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个严谨的医学评估与作答AI，仅输出标准 JSON 格式数据。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,       # 按照要求设置为 0
            max_tokens=256,        # 按照要求设置为 256
            frequency_penalty=0.0,
            top_p=1.0
        )

        raw_result = response.choices[0].message.content.strip()

        # ================== 1. 剥离可能的  comprehend 思考过程 ==================
        if " comprehend" in raw_result:
            raw_result = raw_result.split(" comprehend")[-1].strip()

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
        return {"answer": "生成失败", "reason": "生成失败"}


# ==========================================
# 3. 定义单个模型的完整文件处理工作流
# ==========================================
def process_single_model(config, folder_path):
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
            # 处理原始文件，跳过已经加上 _voted 的文件
            if filename.endswith(".jsonl") and "_voted" not in filename:
                input_file_path = os.path.join(folder_path, filename)
                # 输出文件加上 _voted 后缀
                output_filename = f"{safe_model_name}_{filename.replace('.jsonl', '_voted.jsonl')}"
                output_file_path = os.path.join(output_dir, output_filename)

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
                        opt_a = data.get("A", "")
                        opt_b = data.get("B", "")
                        opt_c = data.get("C", "")
                        opt_d = data.get("D", "")
                        opt_e = data.get("E", "")

                        print(f"  [{model_name}] 正在处理: {filename} -> 第 {idx + 1} 条 (ID: {data_id})")

                        llm_result = None

                        # 引入重试机制
                        for attempt in range(MAX_RETRIES):
                            # 调用 API 获取选项与理由
                            llm_result = answer_medical_question(client, model_name, question, opt_a, opt_b, opt_c, opt_d, opt_e)

                            # 检查 1：返回格式是否错误，或者内部表明生成失败
                            if not isinstance(llm_result, dict) or llm_result.get("answer") in ["生成失败", "生成异常"]:
                                print(f"    [{model_name}] 生成异常，重试 {attempt + 1}/{MAX_RETRIES}...")
                                continue

                            # 检查 2：提取出来的 answer 必须是 A, B, C, D, E 中的一个
                            ans = str(llm_result.get("answer", "")).strip().upper()
                            if ans not in ["A", "B", "C", "D", "E"]:
                                print(f"    [{model_name}] 未输出合法选项(当前为: {ans})，重试 {attempt + 1}/{MAX_RETRIES}...")
                                continue
                            else:
                                # 标准化大写赋值回去，防止模型输出小写的 a,b,c
                                llm_result["answer"] = ans

                            # 如果上述检查都通过了，说明作答成功，跳出重试循环
                            break
                        else:
                            # 兜底逻辑
                            print(f"    [{model_name}] ❌ 达到最大重试次数，ID: {data_id} 作答异常。")
                            if not isinstance(llm_result, dict):
                                llm_result = {"answer": "N/A", "reason": "多次重试后依旧异常"}

                        output_data = {
                            "id": data_id,
                            "question": question,
                            "A": opt_a,
                            "B": opt_b,
                            "C": opt_c,
                            "D": opt_d,
                            "E": opt_e,
                            "answer": llm_result.get("answer", ""),
                            "reason": llm_result.get("reason", "")
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
        print(f"准备在 {FOLDER_PATH} 下并发启动 {len(MODELS_CONFIG)} 个模型的智能作答任务...\n")

        # 使用 ThreadPoolExecutor 实现并行，最大线程数设定为模型的数量
        with ThreadPoolExecutor(max_workers=len(MODELS_CONFIG)) as executor:
            # 提交所有的模型任务到线程池
            futures = [executor.submit(process_single_model, config, FOLDER_PATH) for config in MODELS_CONFIG]

            # 等待并捕获每个线程的最终返回结果
            for future in as_completed(futures):
                result_message = future.result()
                print(f"\n{result_message}")

        print("\n🏆 所有并发模型作答投票任务已全部结束！")