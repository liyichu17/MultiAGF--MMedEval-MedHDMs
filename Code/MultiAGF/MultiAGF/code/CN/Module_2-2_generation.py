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
def generate_hallucination(question, answer):
    """
    根据给定的医学问答，微妙地修改答案生成高可信度的幻觉数据。
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
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个严格执行医学幻觉生成任务的AI，仅输出标准 JSON 格式数据。"},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,  # Temperature = 1
            max_tokens=256,  # Token = 256
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
        return {"Hallucinated_answer": "生成失败", "change_made": "生成失败"}


# ==========================================
# 3. 遍历文件夹处理所有数据并保存
# ==========================================
folder_path = os.path.join("..", "..", "data", "intermediate_data", "combined_subset")
output_dir = os.path.join("..", "..", "data", "intermediate_data", "hallucinated_answers")

# 确保输出目录存在，防止保存时报错
os.makedirs(output_dir, exist_ok=True)

print(f"正在扫描 {folder_path} 文件夹，并开始批量生成医学幻觉数据，请稍候...\n")

if not os.path.exists(folder_path):
    print(f"错误：找不到文件夹 '{folder_path}'。")
else:
    for filename in os.listdir(folder_path):
        # 仅处理 jsonl 文件，并且跳过已经处理过的文件，避免重复套娃
        if filename.endswith(".jsonl") and not filename.endswith("_hallucinated.jsonl"):
            input_file_path = os.path.join(folder_path, filename)
            output_filename = filename.replace('.jsonl', '_hallucinated.jsonl')
            output_file_path = os.path.join(output_dir, output_filename)

            print(f"--- 开始处理文件: {filename} ---")

            try:
                with open(input_file_path, 'r', encoding='utf-8') as infile, \
                        open(output_file_path, 'w', encoding='utf-8') as outfile:

                    for idx, line in enumerate(infile):
                        if not line.strip(): continue
                        data = json.loads(line)

                        data_id = data.get("id", "")  # 提取 id 字段
                        question = data.get("question", "")
                        answer = data.get("answer", "")

                        print(f"  [数据进度] 正在处理第 {idx + 1} 条数据 (ID: {data_id})...")

                        llm_result = None

                        # 引入重试机制
                        for attempt in range(MAX_RETRIES):
                            # 调用 API 获取幻觉数据
                            llm_result = generate_hallucination(question, answer)

                            # 检查 1：返回格式是否错误，或者内部表明生成失败
                            if not isinstance(llm_result, dict) or llm_result.get("Hallucinated_answer") in ["生成失败",
                                                                                                             "生成异常"]:
                                print(f"    [警告] 生成异常，正在进行第 {attempt + 1}/{MAX_RETRIES} 次重试...")
                                continue

                            hallucinated_ans = llm_result.get("Hallucinated_answer", "")

                            # 检查 2：生成的幻觉答案是否与原答案完全一样（去除首尾空格后比较）
                            if hallucinated_ans.strip() == answer.strip():
                                print(
                                    f"    [警告] 幻觉答案与原答案完全相同，正在进行第 {attempt + 1}/{MAX_RETRIES} 次重试...")
                                continue

                            # 检查 3：字数约束检查 (可选补充：如果模型偶尔越界，可要求重试。这里仅作轻度提示，不强制失败，以免死循环)
                            if len(hallucinated_ans) > 140:
                                print(f"    [提示] 幻觉答案字数({len(hallucinated_ans)})超过140字，但仍予以保留。")

                            # 如果上述关键检查都通过了，说明生成成功，跳出重试循环
                            break
                        else:
                            # 如果达到了最大重试次数仍然没有 break，则执行此处的兜底逻辑
                            print(f"    [错误] 达到最大重试次数 ({MAX_RETRIES})，该条数据生成质量不佳。")
                            if not isinstance(llm_result, dict):
                                llm_result = {"Hallucinated_answer": "生成异常", "change_made": "多次重试后依旧异常"}

                        # 按照要求组装最终输出格式，包含 id
                        output_data = {
                            "id": data_id,
                            "question": question,
                            "answer": answer,
                            "Hallucinated_answer": llm_result.get("Hallucinated_answer", ""),
                            "change_made": llm_result.get("change_made", "")
                        }

                        # 写入新的 jsonl 文件
                        outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')

                print(f"✅ 文件 {filename} 处理完成，结果已保存至: {output_filename}\n")

            except Exception as e:
                print(f"读取或写入文件 {filename} 时发生错误: {e}")

print("全部任务执行完毕！")