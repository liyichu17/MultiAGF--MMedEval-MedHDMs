import subprocess
import os
import sys

def run_module(module_name, config_json=None, base_file_json=None):
    """
    运行指定的 Python 脚本模块，并等待其完成。
    如果模块运行成功（退出码为 0），则返回 True；否则返回 False。
    """
    print(f"\n{'='*20} 正在启动 {module_name} {'='*20}")

    # 确保在当前脚本目录下运行对应的 Module 文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(current_dir, module_name)

    if not os.path.exists(module_path):
        print(f"错误：未找到文件 {module_path}")
        return False

    try:
        # 使用 sys.executable 确保使用相同的 Python 解释器
        # subprocess.run 会阻塞直到命令执行完成
        cmd = [sys.executable, module_path]
        if config_json:
            cmd.append(config_json)
        if base_file_json:
            cmd.append(base_file_json)

        result = subprocess.run(cmd, check=True)
        print(f"✅ {module_name} 运行成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {module_name} 运行出错，退出码: {e.returncode}")
        return False
    except Exception as e:
        print(f"发生未知错误：{e}")
        return False

def main():
    # 模型配置信息
    models_config = [
        {"model_name": "Model-A", "api_key": "YOUR_API_KEY_1", "base_url": "https://api.model1.com/v1"},
        {"model_name": "Model-B", "api_key": "YOUR_API_KEY_2", "base_url": "https://api.model2.com/v1"},
        {"model_name": "Model-C", "api_key": "YOUR_API_KEY_3", "base_url": "https://api.model3.com/v1"},
        {"model_name": "Model-D", "api_key": "YOUR_API_KEY_4", "base_url": "https://api.model4.com/v1"},
        {"model_name": "Model-E", "api_key": "YOUR_API_KEY_5", "base_url": "https://api.model5.com/v1"},
    ]
    # 将模型配置转换为 JSON 字符串作为参数传递
    import json
    config_json = json.dumps(models_config)

    # 基础文件名配置信息（用于 Module 3 和 Module 7）
    base_file_config = {
        "base_filename": "MedBench3000_cl_hallucinated.jsonl"
    }
    base_file_json = json.dumps(base_file_config)

    # 按顺序执行的模块列表
    modules_to_run = [
        "Module_1_preprocessing.py",
        "Module_2_multi_generation.py",
        "Module_3_voting_pool_creation.py",
        "Module_4_multi_voting.py",
        "Module_5_voting_consolidation.py",
        "Module_6_determination.py",
        "Module_7_dataset_formation.py"
    ]

    print("开始串行执行所有项目模块...")

    for module in modules_to_run:
        # Module 3 和 Module 7 需要多传一个基础文件名的参数
        if module in ["Module_3_voting_pool_creation.py", "Module_7_dataset_formation.py"]:
            success = run_module(module, config_json, base_file_json)
        else:
            success = run_module(module, config_json)
        
        if not success:
            print(f"\n停止运行：由于 {module} 出现错误，后续模块已取消启动。")
            sys.exit(1)

    print(f"\n{'='*20} 所有模块均已按顺序顺利执行完毕！ {'='*20}")

if __name__ == "__main__":
    main()
