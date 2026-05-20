# MultiAGF: A Multi-Agent Consensus Framework for Adversarial Medical Hallucination Synthesis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework: MultiAGF](https://img.shields.io/badge/Framework-MultiAGF-green.svg)]()

This repository contains the official codebase and benchmark artifacts for **MultiAGF (Multi-Agent Consensus Framework)**, an advanced pipeline designed for automated synthesis, evaluation, and consensus-driven determination of high-fidelity, adversarial medical hallucination datasets. 

---

## 📢 Code, Data, and Model Weights / 论文代码、数据与权重获取

### English
The source code and processed evaluation data for our paper are organized and uploaded as described below. If you require the corresponding model weights for **MedHDMs** (the foundational base models fine-tuned via our specialized medical CoT trajectories), please visit the following link:
🔗 **[https://drive.google.com/drive/folders/17QpcCZ_iaEYoJkE245fnmX2NmtGwpm_b?usp=drive_link](https://drive.google.com/drive/folders/17QpcCZ_iaEYoJkE245fnmX2NmtGwpm_b?usp=drive_link)**

### 中文
论文的代码及数据已按照下文所述架构上传并归档。如果您的研究需要 **MedHDMs** 的相关模型权重数据（基于我们定制的医学思维链 CoT 推理路径微调的基座大模型权重），可以访问以下链接进行下载：
🔗 **[https://drive.google.com/drive/folders/17QpcCZ_iaEYoJkE245fnmX2NmtGwpm_b?usp=drive_link](https://drive.google.com/drive/folders/17QpcCZ_iaEYoJkE245fnmX2NmtGwpm_b?usp=drive_link)**

---

## 📂 Repository Structure & Pipeline Modules / 代码库架构与模块说明

The computational pipeline of the MultiAGF framework is structured into seven core functional modules, tracking the entire data life cycle from raw ingestion to final dataset encapsulation.

### Module 1: Data Preprocessing
* **`preprocessing.py`**: Standardizes heterogeneous raw data formats and filters out redundant fields.
* **`1_full_preprocessing.py`**: Integrates baseline cleansing functions with specific data processing rules customized for the **MMedBench** dataset.
* **`1-2_full_preprocessing.py`**: Tailored for the open-ended text structure of the **Huatuo** dataset.

### Module 2: Automated Hallucination Generation
* **`generation.py`**: Drives single-agent LLM invocations via API interfaces to synthesize initial adversarial hallucination candidates.
* **`multi_generation.py`** (Module 2 multi): The advanced multi-agent deployment version. Natively establishes a parallelized multi-threading concurrent environment invoking 5 heterogeneous LLMs (Gemini, DeepSeek, Kimi, ERNIE, ChatGPT) to systematically introduce deep-seated cognitive divergence.

### Module 3: Voting Pool Creation
* **`voting_pool_creation.py`**: Aligns, structures, and aggregates the multi-agent outputs, compiling them into a unified "voting pool" (`Pool_data.jsonl`) to support downstream consensus evaluation.

### Module 4: Consensus Voting
* **`multi_voting.py`**: Orchestrates automated, concurrent multi-agent voting via APIs. Deploys the LLM ensemble to independently evaluate the medical plausibility and unfaithfulness of candidate hallucinations. `voting.py` provides the single-model voting baseline.

### Module 5: Voting Consolidation
* **`voting_consolidation.py`**: Collects independent agent decisions and parses them into a quantified consensus matrix, providing empirical metadata and inter-rater statistical metrics (`model_answer_count1_1.xlsx`).

### Module 6: Final Determination
* **`determination.py`**: Executes the core **Weighted Consensus Rule**. Resolves edge cases and mathematically mitigates the "Kappa Paradox" via advanced reliability indexing, rendering optimized selections (`model_answer_count2_1.xlsx`) and isolating the highest-quality hallucination instances (`*_final.jsonl`).

### Module 7: Dataset Formation
* **`dataset_formation.py`**: Formats, aligns, and pairs the ground-truth medical query-response combinations with the optimally selected hallucinations to yield the standardized final benchmark dataset (`final_hal_data.jsonl`).

---

## 🛠️ Base Models for MedHDMs / 基座模型选择说明

To ensure seamless on-premise private deployment within healthcare environments under localized GPU memory limitations (VRAM constraints of 24GB/32GB), the base models utilized for MedHDMs are strictly bounded within the **7B–9B parameter range** and adhere to highly permissive licenses (e.g., Apache 2.0, MIT).

| Base Model Name | Parameters | Primary License Type | Architectural Heterogeneity Feature |
| :--- | :--- | :--- | :--- |
| **DeepSeek-LLM-7B** | 7B | MIT | Highly cost-effective foundational base |
| **Qwen3.5-9B** | 9B | Apache 2.0 | Cutting-edge architecture iteration contrast |
| **Qwen3-8B** | 8B | Apache 2.0 | Baseline benchmark for ancestral ablation |
| **LLaMA3-Chinese-8B-Instruct** | 8B | Custom (Permissive) | Cross-lingual domain generalization variant |
| **GLM4-9B** | 9B | Custom (Commercial Friendly) | Registration-based open commercial usage |

---

## 🚀 Getting Started / 快速启动

### 1. Installation
Clone this repository and set up the computational environment:
```bash
git clone [https://github.com/your-username/MultiAGF.git](https://github.com/your-username/MultiAGF.git)
cd MultiAGF
pip install -r requirements.txt