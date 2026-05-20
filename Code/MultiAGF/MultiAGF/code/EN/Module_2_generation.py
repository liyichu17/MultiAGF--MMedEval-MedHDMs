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
def generate_hallucination(question, answer, rationale):
    """
    Based on the given Q&A and explanation of the underlying principles, generate misleading medical hallucination answers and modification notes.
    """
    prompt = f"""
任务描述：
1. 你是一名专门用于生成高可信医学幻觉答案的模型。
2. 在原有 "answer" 的基础上，微妙地修改答案，使其变为幻觉答案。该答案需保持表面上作为“有效答案”的合理性，但当仔细阅读 "rationale" 字段时会发现该幻觉答案在事实上的错误。
3. 请严格输出一个合法的 JSON 字典，绝对不要输出任何多余的分析文本或 Markdown 标记。
4. 【重要约束】你生成的幻觉答案绝对不能与原始答案一模一样！必须产生实质性的医学事实错误。

输入数据：
[question]: {question}
[answer]: {answer}
[rationale]: {rationale}

请严格按照以下 JSON 格式输出结果：
{{
  "Hallucinated_answer": "你的幻觉答案",
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
            temperature=1.0,  # Enhancing randomness and creativity to generate diverse hallucinations
            max_tokens=256,  # Set to 256 as required
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
        return {"Hallucinated_answer": "生成失败", "change_made": "生成失败"}


# ==========================================
# 3. Iterate through the folder to process all data and save it
# ==========================================
folder_path = os.path.join("..", "..", "data", "intermediate_data", "combined_subset")
output_dir = os.path.join("..", "..", "data", "intermediate_data", "hallucinated_answers")

print(f"Scanning the {folder_path} folder and starting to generate medical hallucination data in bulk. Please wait....\n")

if not os.path.exists(folder_path):
    print(f"Error: Folder not found '{folder_path}'。")
else:
    for filename in os.listdir(folder_path):
        # Process only JSONL files and skip those that have already been processed to avoid nested processing.
        if filename.endswith(".jsonl") and not filename.endswith("_hallucinated.jsonl"):
            input_file_path = os.path.join(folder_path, filename)
            output_filename = filename.replace('.jsonl', '_hallucinated.jsonl')
            output_file_path = os.path.join(output_dir, output_filename)

            print(f"--- Start processing the file: {filename} ---")

            try:
                with open(input_file_path, 'r', encoding='utf-8') as infile, \
                        open(output_file_path, 'w', encoding='utf-8') as outfile:

                    for idx, line in enumerate(infile):
                        if not line.strip(): continue
                        data = json.loads(line)

                        data_id = data.get("id", "")  # Extract the “id” field
                        question = data.get("question", "")
                        answer = data.get("answer", "")
                        rationale = data.get("rationale", "")

                        print(f"  [Data progress] Processing the {idx + 1} th data record (ID: {data_id})...")

                        llm_result = None

                        # Implementing a retry mechanism
                        for attempt in range(MAX_RETRIES):
                            # Call the API to retrieve hallucination data
                            llm_result = generate_hallucination(question, answer, rationale)

                            # Check 1: Check whether the return format is incorrect, or whether there is an internal indication that generation has failed
                            if not isinstance(llm_result, dict) or llm_result.get("Hallucinated_answer") in ["生成失败",
                                                                                                             "生成异常"]:
                                print(f"    [Warning] An exception has been raised; this is the {attempt + 1}/{MAX_RETRIES} th retry...")
                                continue

                            hallucinated_ans = llm_result.get("Hallucinated_answer", "")

                            # Check 2: Does the generated answer match the original answer exactly (after removing leading and trailing spaces)?
                            if hallucinated_ans.strip() == answer.strip():
                                print(
                                    f"    [Warning] The generated answer is identical to the original answer; this is the {attempt + 1}/{MAX_RETRIES} th retry...")
                                continue

                            # If all the above checks are passed, this indicates that the generation was successful; exit the retry loop
                            break
                        else:
                            # If the maximum number of retries has been reached and there is still no break, execute the fallback logic here
                            print(f"    [Error] The maximum number of retries ({MAX_RETRIES}) has been reached; the data generated for this record is of poor quality.")
                            if not isinstance(llm_result, dict):
                                llm_result = {"Hallucinated_answer": "生成异常", "change_made": "多次重试后依旧异常"}

                        # Assemble the final output format as required, including the ID
                        output_data = {
                            "id": data_id,
                            "question": question,
                            "answer": answer,
                            "Hallucinated_answer": llm_result.get("Hallucinated_answer", ""),
                            "change_made": llm_result.get("change_made", "")
                        }

                        # Write to a new JSONL file
                        outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')

                print(f"✅ The file {filename} has been processed, and the results have been saved to: {output_filename}\n")

            except Exception as e:
                print(f"An error occurred whilst reading or writing the file {filename} : {e}")

print("All tasks have been completed!")