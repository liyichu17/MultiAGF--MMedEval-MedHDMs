import os
import json
import re
import sys
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. Configure the individual parameters (API, URL, model name) for five LLMs
# ==========================================
# Default configuration; if not passed via parameters, use this configuration
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
    # }  # Add as many client lists as there are models
]

# If there is a command-line argument passed, override the default configuration
if len(sys.argv) > 1:
    try:
        MODELS_CONFIG = json.loads(sys.argv[1])
        print("💡 The model configuration has been loaded via external parameters.")
    except Exception as e:
        print(f"⚠️ Failed to parse external parameters, using built-in configuration: {e}")

MAX_RETRIES = 5  # Set the maximum number of retries, this can be adjusted as required
FOLDER_PATH = os.path.join("..", "..", "data", "intermediate_data", "combined_subset")
output_dir = os.path.join("..", "..", "data", "intermediate_data", "hallucinated_answers")

# ==========================================
# 2. Define the underlying logic for a single API request
# ==========================================
def generate_hallucination(client, model_name, question, answer, rationale):
    """
    Based on the given Q&A and explanation of the underlying principles, a specific client is invoked to generate a simulated answer.
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

        # ================== 1. Isolating the potential  comprehend thought process ==================
        if " comprehend" in raw_result:
            raw_result = raw_result.split(" comprehend")[-1].strip()

        # ================== 2. Extracting JSON structures using regular expressions ==================
        clean_json_str = re.sub(r"```json|```", "", raw_result).strip()
        json_match = re.search(r'\{.*?\}', clean_json_str, re.DOTALL)
        if json_match:
            clean_json_str = json_match.group(0)

        # ================== 3. Parsing JSON ==================
        try:
            return json.loads(clean_json_str)
        except json.JSONDecodeError:
            # Handling cases where the right curly bracket is missing
            if not clean_json_str.endswith("}"):
                try:
                    return json.loads(clean_json_str + "}")
                except:
                    pass
            raise ValueError(f"JSON parsing failed completely: {clean_json_str}")

    except Exception as e:
        print(f"[{model_name}] API call or parsing failed: {e}")
        return {"Hallucinated_answer": "生成失败", "change_made": "生成失败"}


# ==========================================
# 3. Defining a complete file-handling workflow for a single model
# ==========================================
def process_single_model(config, folder_path):
    """
    This function will run in a separate thread and is responsible for all data processing for a specific model.
    """
    model_name = config["model_name"]
    api_key = config["api_key"]
    base_url = config["base_url"]

    # Instantiate a dedicated OpenAI Client for the current model
    client = OpenAI(api_key=api_key, base_url=base_url)

    safe_model_name = model_name.replace('/', '_').replace(':', '_')

    print(f"🚀 [Thread started] Model: {model_name} is ready; starting to scan files...")

    try:
        for filename in os.listdir(folder_path):
            # Process only JSONL files and skip those that have already been processed to avoid nested processing.
            if filename.endswith(".jsonl") and "_hallucinated" not in filename:
                input_file_path = os.path.join(folder_path, filename)
                output_filename = f"{safe_model_name}_{filename.replace('.jsonl', '_hallucinated.jsonl')}"
                output_file_path = os.path.join(output_dir, output_filename)

                if os.path.exists(output_file_path):
                    print(f"⚠️ The file {output_filename} for [{model_name}] already exists; skipping.")
                    continue

                with open(input_file_path, 'r', encoding='utf-8') as infile, \
                        open(output_file_path, 'w', encoding='utf-8') as outfile:

                    for idx, line in enumerate(infile):
                        if not line.strip(): continue
                        data = json.loads(line)

                        data_id = data.get("id", "")
                        question = data.get("question", "")
                        answer = data.get("answer", "")
                        rationale = data.get("rationale", "")

                        print(f"  [{model_name}] Processing: {filename} -> Record {idx + 1} (ID: {data_id})")

                        llm_result = None

                        # Implementing a retry mechanism
                        for attempt in range(MAX_RETRIES):
                            # Call the API to retrieve hallucination data
                            llm_result = generate_hallucination(client, model_name, question, answer, rationale)

                            # Check 1: Check whether the return format is incorrect, or whether there is an internal indication that generation has failed
                            if not isinstance(llm_result, dict) or llm_result.get("Hallucinated_answer") in ["生成失败",
                                                                                                             "生成异常"]:
                                print(f"    [{model_name}] generated an exception; retrying {attempt + 1}/{MAX_RETRIES}...")
                                continue

                            hallucinated_ans = llm_result.get("Hallucinated_answer", "")

                            # Check 2: Does the generated answer match the original answer exactly (after removing leading and trailing spaces)?
                            if hallucinated_ans.strip() == answer.strip():
                                print(f"    [{model_name}] The answer has not been substantially modified; retrying {attempt + 1}/{MAX_RETRIES}...")
                                continue

                            # If all the above checks are passed, this indicates that the generation was successful; exit the retry loop
                            break
                        else:
                            print(f"    [{model_name}] ❌ Maximum retry limit reached; ID: {data_id} – poor-quality output generated.")
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

                print(f"✅ The file {filename} for [{model_name}] has been processed!")

    except Exception as e:
        return f"❌ A fatal error has occurred in the [{model_name}] thread: {e}"

    return f"🎉 All tasks for [{model_name}] have been successfully completed!"


# ==========================================
# 4. Main programme: Initiate multi-threaded parallel execution
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(FOLDER_PATH):
        print(f"Error: Folder not found '{FOLDER_PATH}'。")
    else:
        print(f"Preparing to launch {len(MODELS_CONFIG)} model tasks concurrently under {FOLDER_PATH}...\n")

        # Use ThreadPoolExecutor to implement parallel processing, setting the maximum number of threads to the number of models
        with ThreadPoolExecutor(max_workers=len(MODELS_CONFIG)) as executor:
            # Submit all model tasks to the thread pool
            futures = [executor.submit(process_single_model, config, FOLDER_PATH) for config in MODELS_CONFIG]

            # Wait for and capture the final return value of each thread
            for future in as_completed(futures):
                result_message = future.result()
                print(f"\n{result_message}")

        print("\n🏆 All tasks for generating concurrency models have now been completed!")