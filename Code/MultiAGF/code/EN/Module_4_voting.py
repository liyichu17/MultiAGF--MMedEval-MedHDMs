import os
import json
import re
from openai import OpenAI

# ==========================================
# 1. Configuring the LLMs API client
# ==========================================
API_KEY = "YOUR_API_KEY"  # Replace this with your API_KEY
BASE_URL = "https://api.model.com/v1"  # Replace with your local or cloud API endpoint
MODEL_NAME = "Model"  # Replace with the actual LLMs name

MAX_RETRIES = 5  # Set the maximum number of retries, this can be adjusted as required

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


# ==========================================
# 2. Defining LLMs requests and Prompt Templates
# ==========================================
def answer_medical_question(question, opt_a, opt_b, opt_c, opt_d, opt_e):
    """
    Given a medical question and five options, ask the large language model to reason and select the correct answer along with its justification.
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
            temperature=0.0,
            max_tokens=256,
            frequency_penalty=0.0,
            top_p=1.0
        )

        raw_result = response.choices[0].message.content.strip()

        # ================== 1. Isolating the potential <think> thought process ==================
        if "</think>" in raw_result:
            raw_result = raw_result.split("</think>")[-1].strip()

        # ================== 2. Extracting JSON structures using regular expressions ==================
        clean_json_str = re.sub(r"```json|```", "", raw_result).strip()
        json_match = re.search(r'\{.*?\}', clean_json_str, re.DOTALL)
        if json_match:
            clean_json_str = json_match.group(0)

        # ================== 3. Parsing JSON ==================
        try:
            parsed_data = json.loads(clean_json_str)
            return parsed_data
        except json.JSONDecodeError:
            # Handling cases where the right curly bracket is missing
            if not clean_json_str.endswith("}"):
                try:
                    return json.loads(clean_json_str + "}")
                except:
                    pass
            raise ValueError(f"JSON parsing failed completely: {clean_json_str}")

    except Exception as e:
        print(f"An error occurred whilst calling a large model or parsing JSON: {e}")
        return {"answer": "生成失败", "reason": "生成失败"}


# ==========================================
# 3. Iterate through the folder to process all data and save it
# ==========================================
folder_path = os.path.join("..", "..", "data", "intermediate_data", "voting_pool")
output_dir = os.path.join("..", "..", "data", "intermediate_data", "voting_results")

print(f"Scanning the {folder_path} folder and starting to generate medical hallucination data in bulk. Please wait...\n")

if not os.path.exists(folder_path):
    print(f"Error: Folder not found '{folder_path}'。")
else:
    for filename in os.listdir(folder_path):
        # Process only JSONL files and skip those that have already been processed to avoid nested processing.
        if filename.endswith(".jsonl") and not filename.endswith("_voted.jsonl"):
            input_file_path = os.path.join(folder_path, filename)
            # Add the suffix _voted to the output file
            output_filename = filename.replace('.jsonl', '_voted.jsonl')
            output_file_path = os.path.join(output_dir, output_filename)

            print(f"--- 📂 Start processing the file: {filename} ---")

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

                        print(f"  [Data progress] Processing the {idx + 1} th data record (ID: {data_id})...")

                        llm_result = None

                        # Implementing a retry mechanism
                        for attempt in range(MAX_RETRIES):
                            # Call the API to retrieve hallucination data
                            llm_result = answer_medical_question(question, opt_a, opt_b, opt_c, opt_d, opt_e)

                            # Check 1: Check whether the return format is incorrect, or whether the internal status indicates a generation failure
                            if not isinstance(llm_result, dict) or llm_result.get("answer") in ["生成失败", "生成异常"]:
                                print(f"    [Warning] An exception has been raised; this is the {attempt + 1}/{MAX_RETRIES} th retry...")
                                continue

                            # Check 2: The extracted answer must be one of A, B, C, D or E
                            ans = str(llm_result.get("answer", "")).strip().upper()
                            if ans not in ["A", "B", "C", "D", "E"]:
                                print(f"    [Warning] The model has not returned a valid option (current output is: {ans}). This is the {attempt + 1}th retry out of {MAX_RETRIES}....")
                                continue
                            else:
                                # Standardise capitalised assignments to prevent the model from outputting lowercase a, b and c
                                llm_result["answer"] = ans

                            # If all the above checks are passed, this indicates that the response was successful, and the retry loop is exited
                            break
                        else:
                            # The underlying logic
                            print(f"    [Error] The maximum number of retries ({MAX_RETRIES}) has been reached; an error occurred whilst processing this record.")
                            if not isinstance(llm_result, dict):
                                llm_result = {"answer": "N/A", "reason": "多次重试后依旧异常"}

                        # Assemble the final output format as required
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

                        # Write to a new JSONL file
                        outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')

                print(f"✅ he file {filename} has been processed, and the results have been saved to: {output_filename}\n")

            except Exception as e:
                print(f"An error occurred whilst reading or writing the file {filename} : {e}")

print("🎉 All tasks have been completed!")