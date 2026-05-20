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
FOLDER_PATH = os.path.join("..", "..", "data", "intermediate_data", "voting_pool")
output_dir = os.path.join("..", "..", "data", "intermediate_data", "voting_results")

# ==========================================
# 2. Define the underlying logic for individual API requests (submitting answers and voting)
# ==========================================
def answer_medical_question(client, model_name, question, opt_a, opt_b, opt_c, opt_d, opt_e):
    """
    Given a medical question and five options, ask the large language model to reason through the options and select the correct answer along with its justification.
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
            temperature=0.0,
            max_tokens=256,
            frequency_penalty=0.0,
            top_p=1.0
        )

        raw_result = response.choices[0].message.content.strip()

        # ================== 1. Isolating the potential  ground thought process ==================
        if "地" in raw_result:
            raw_result = raw_result.split("地")[-1].strip()

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
        return {"answer": "生成失败", "reason": "生成失败"}


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
            # Process the original files, skipping those already marked with _voted
            if filename.endswith(".jsonl") and "_voted" not in filename:
                input_file_path = os.path.join(folder_path, filename)
                # Add the suffix _voted to the output file
                output_filename = f"{safe_model_name}_{filename.replace('.jsonl', '_voted.jsonl')}"
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
                        opt_a = data.get("A", "")
                        opt_b = data.get("B", "")
                        opt_c = data.get("C", "")
                        opt_d = data.get("D", "")
                        opt_e = data.get("E", "")

                        print(f"  [{model_name}] Processing: {filename} -> Record {idx + 1} (ID: {data_id})")

                        llm_result = None

                        # Implementing a retry mechanism
                        for attempt in range(MAX_RETRIES):
                            # Call the API to retrieve hallucination data
                            llm_result = answer_medical_question(client, model_name, question, opt_a, opt_b, opt_c, opt_d, opt_e)

                            # Check 1: Check whether the return format is incorrect, or whether the internal status indicates a generation failure
                            if not isinstance(llm_result, dict) or llm_result.get("answer") in ["生成失败", "生成异常"]:
                                print(f"    [{model_name}] generated an exception; retrying {attempt + 1}/{MAX_RETRIES}...")
                                continue

                            # Check 2: The extracted answer must be one of A, B, C, D or E
                            ans = str(llm_result.get("answer", "")).strip().upper()
                            if ans not in ["A", "B", "C", "D", "E"]:
                                print(f"    [{model_name}] did not return a valid option (current: {ans}); retrying {attempt + 1}/{MAX_RETRIES}...")
                                continue
                            else:
                                # Standardise capitalised assignments to prevent the model from outputting lowercase a, b and c
                                llm_result["answer"] = ans

                            # If all the above checks are passed, this indicates that the response was successful, and the retry loop is exited
                            break
                        else:
                            # The underlying logic
                            print(f"    [{model_name}] ❌ Maximum retry limit reached; error occurred whilst answering for ID: {data_id}.")
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
        print(f"Preparing to launch {len(MODELS_CONFIG)} intelligent answering tasks for the models concurrently under {FOLDER_PATH}...\n")

        # Use ThreadPoolExecutor to implement parallel processing, setting the maximum number of threads to the number of models
        with ThreadPoolExecutor(max_workers=len(MODELS_CONFIG)) as executor:
            # Submit all model tasks to the thread pool
            futures = [executor.submit(process_single_model, config, FOLDER_PATH) for config in MODELS_CONFIG]

            # Wait for and capture the final return value of each thread
            for future in as_completed(futures):
                result_message = future.result()
                print(f"\n{result_message}")

        print("\n🏆 All tasks for generating concurrency models have now been completed!")