import subprocess
import os
import sys

def run_module(module_name, config_json=None, base_file_json=None):
    """
    Runs the specified Python script module and waits for it to finish.
    If the module runs successfully (with an exit code of 0), it returns True; otherwise, it returns False.
    """
    print(f"\n{'='*20} Starting {module_name} {'='*20}")

    # Ensure that the relevant module file is run from the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(current_dir, module_name)

    if not os.path.exists(module_path):
        print(f"Error: File not found {module_path}")
        return False

    try:
        # Use `sys.executable` to ensure the same Python interpreter is used
        # `subprocess.run` will block until the command has finished executing
        cmd = [sys.executable, module_path]
        if config_json:
            cmd.append(config_json)
        if base_file_json:
            cmd.append(base_file_json)

        result = subprocess.run(cmd, check=True)
        print(f"✅ {module_name} Execution successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {module_name} Runtime error, exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"An unknown error has occurred：{e}")
        return False

def main():
    # Model configuration information
    models_config = [
        {"model_name": "Model-A", "api_key": "YOUR_API_KEY_1", "base_url": "https://api.model1.com/v1"},
        {"model_name": "Model-B", "api_key": "YOUR_API_KEY_2", "base_url": "https://api.model2.com/v1"},
        {"model_name": "Model-C", "api_key": "YOUR_API_KEY_3", "base_url": "https://api.model3.com/v1"},
        {"model_name": "Model-D", "api_key": "YOUR_API_KEY_4", "base_url": "https://api.model4.com/v1"},
        {"model_name": "Model-E", "api_key": "YOUR_API_KEY_5", "base_url": "https://api.model5.com/v1"},
    ]
    # Convert model configuration to a JSON string to pass as an argument
    import json
    config_json = json.dumps(models_config)

    # Base filename configuration info (used in Module 3 and Module 7)
    base_file_config = {
        "base_filename": "MedBench3000_cl_hallucinated.jsonl"
    }
    base_file_json = json.dumps(base_file_config)

    # List of modules to be executed in sequence
    modules_to_run = [
        "Module_1_preprocessing.py",
        "Module_2_multi_generation.py",
        "Module_3_voting_pool_creation.py",
        "Module_4_multi_voting.py",
        "Module_5_voting_consolidation.py",
        "Module_6_determination.py",
        "Module_7_dataset_formation.py"
    ]

    print("Start executing all project modules sequentially...")

    for module in modules_to_run:
        # Module 3 and Module 7 need to pass an extra base_filename parameter
        if module in ["Module_3_voting_pool_creation.py", "Module_7_dataset_formation.py"]:
            success = run_module(module, config_json, base_file_json)
        else:
            success = run_module(module, config_json)

        if not success:
            print(f"\nStopped: Due to an error in {module}, subsequent modules have been prevented from starting.")
            sys.exit(1)

    print(f"\n{'='*20} All modules have been successfully completed in sequence! {'='*20}")

if __name__ == "__main__":
    main()
