import os
import json
import re
from openai import OpenAI

# ==========================================
# 1. 配置大模型 API 客户端
# ==========================================
API_KEY = "YOUR_API_KEY"  # 替换为你的 API_KEY
BASE_URL = "https://api.model.com/v1"  # 替换为你的本地或云端 API 接口
MODEL_NAME = "Model"  # 替换为实际的模型名称

MAX_RETRIES = 5  # 设置最大重试次数，可按需调节

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


# ==========================================
# 2. 定义大模型请求与 Prompt 模板
# ==========================================
def answer_medical_question(question, opt_a, opt_b, opt_c, opt_d, opt_e):
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
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个严谨的医学评估与作答AI，仅输出标准 JSON 格式数据。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,       # 按照要求设置为 0
            max_tokens=256,        # 按照要求设置为 256
            frequency_penalty=0.0, # 按照要求设置为 0
            top_p=1.0              # 按照要求设置为 1
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
            parsed_data = json.loads(clean_json_str)
            return parsed_data
        except json.JSONDecodeError:
            # 抢救漏掉右大括号的情况
            if not clean_json_str.endswith("}"):
                try:
                    return json.loads(clean_json_str + "}")
                except:
                    pass
            raise ValueError(f"JSON解析彻底失败: {clean_json_str}")

    except Exception as e:
        print(f"调用大模型或解析 JSON 时出错: {e}")
        return {"answer": "生成失败", "reason": "生成失败"}


# ==========================================
# 3. 遍历文件夹处理所有数据并保存
# ==========================================
folder_path = os.path.join("..", "..", "data", "intermediate_data", "voting_pool")
output_dir = os.path.join("..", "..", "data", "intermediate_data", "voting_results")

print(f"正在扫描 {folder_path} 文件夹，并开始批量进行医学试题智能作答，请稍候...\n")

if not os.path.exists(folder_path):
    print(f"错误：找不到文件夹 '{folder_path}'。")
else:
    for filename in os.listdir(folder_path):
        # 仅处理原始 jsonl 文件，跳过已处理的带 "_voted" 后缀的文件
        if filename.endswith(".jsonl") and not filename.endswith("_voted.jsonl"):
            input_file_path = os.path.join(folder_path, filename)
            # 输出文件加上 _voted 后缀
            output_filename = filename.replace('.jsonl', '_voted.jsonl')
            output_file_path = os.path.join(output_dir, output_filename)

            print(f"--- 📂 开始处理文件: {filename} ---")

            try:
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

                        print(f"  [数据进度] 正在处理第 {idx + 1} 条数据 (ID: {data_id})...")

                        llm_result = None

                        # 引入重试机制
                        for attempt in range(MAX_RETRIES):
                            # 调用 API 获取选项与理由
                            llm_result = answer_medical_question(question, opt_a, opt_b, opt_c, opt_d, opt_e)

                            # 检查 1：返回格式是否错误，或者内部表明生成失败
                            if not isinstance(llm_result, dict) or llm_result.get("answer") in ["生成失败", "生成异常"]:
                                print(f"    [警告] 生成异常，正在进行第 {attempt + 1}/{MAX_RETRIES} 次重试...")
                                continue

                            # 检查 2：提取出来的 answer 必须是 A, B, C, D, E 中的一个
                            ans = str(llm_result.get("answer", "")).strip().upper()
                            if ans not in ["A", "B", "C", "D", "E"]:
                                print(f"    [警告] 模型未输出合法选项(当前输出为: {ans})，正在进行第 {attempt + 1}/{MAX_RETRIES} 次重试...")
                                continue
                            else:
                                # 标准化大写赋值回去，防止模型输出小写的 a,b,c
                                llm_result["answer"] = ans

                            # 如果上述检查都通过了，说明作答成功，跳出重试循环
                            break
                        else:
                            # 兜底逻辑
                            print(f"    [错误] 达到最大重试次数 ({MAX_RETRIES})，该条数据作答异常。")
                            if not isinstance(llm_result, dict):
                                llm_result = {"answer": "N/A", "reason": "多次重试后依旧异常"}

                        # 按照要求组装最终输出格式
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

                print(f"✅ 文件 {filename} 处理完成，结果已保存至: {output_filename}\n")

            except Exception as e:
                print(f"读取或写入文件 {filename} 时发生错误: {e}")

print("🎉 全部任务执行完毕！")