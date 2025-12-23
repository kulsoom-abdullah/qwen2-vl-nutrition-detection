
# Qwen2-VL Nutrition Table Detection System

A production-grade vision-language system for automated nutrition table detection. This project bridges the gap between multimodal research and enterprise deployment, featuring a fine-tuned Qwen2-VL-7B model achieving a **30.7% Mean IoU improvement** over the baseline.

## 🚀 Production Engineering & System Architecture

This repository implements high-availability engineering patterns designed for production environments:

* **Strategy Pattern Inference:** Abstracted the inference layer to support multiple backends (vLLM, Triton, or Local Mocks). This enables **"Shift Left" testing** on standard CPU hardware without GPU overhead.
* **BFF (Backend-for-Frontend) Validation:** A FastAPI service handles image validation, coordinate normalization, and resolution capping (1024px) to protect GPU resources from memory spikes.
* **Robust Data Integrity:** Implemented strict Pydantic schemas with custom `@field_validator` logic to ensure model output integrity and prevent coordinate hallucinations.
* **Reproducible Environment:** Full PEP 517 compliance with pinned dependencies and a multi-stage Dockerfile for slim, secure runtime artifacts.

## 🎯 Project Overview

This project fine-tunes **Qwen2-VL-7B** to detect nutrition tables in product images using **QLoRA (4-bit quantized LoRA)** on RunPod A100 GPUs. The notebook documents three experiments comparing different LoRA configurations and training strategies.

![Prediction Example](images/failure_analysis_exp1a.png)

**Dataset**: [OpenFoodFacts Nutrition Table Detection](https://huggingface.co/datasets/openfoodfacts/nutrition-table-detection) (1,106 training images, 123 test images)

## 📊 Experimental Results & Key Findings

| Experiment | Mean IoU | F1@0.5 | Improvement |
|:-----------|:--------:|:------:|:-----------:|
| **Baseline (Zero-Shot)** | 0.590 | 0.654 | - |
| **Exp 1a: LLM LoRA + Masking** ⭐ | **0.771** | **0.893** | **+30.7%** |
| **Exp 1b: LLM LoRA (No Masking)** | 0.745 | 0.870 | +26.3% |
| **Exp 2: Vision+LLM LoRA + Masking** | 0.748 | 0.863 | +26.8% |

### Key Insights

1. **User-Prompt Masking is critical** - Computing loss only on assistant responses (not user prompts) significantly reduced noise and improved coordinate precision by 3.5%
2. **LLM-only LoRA is optimal** - Tuning the vision encoder provided minimal benefit for this detection task
3. **Efficient fine-tuning** - 4-bit NF4 quantization with LoRA (rank=64) achieved strong results on a single A100 GPU
4. **Production optimization** - FP8 quantization delivers 37% faster inference with zero accuracy loss

## 🛠️ Technical Details

- **Model**: Qwen2-VL-7B-Instruct (7B parameters)
- **Training**: QLoRA (4-bit NF4 quantization, LoRA rank=64, α=16)
- **Hardware**: RunPod A100 (80GB VRAM)
- **Framework**: Hugging Face TRL, PEFT, transformers
- **Training time**: ~2 hours per experiment (7 epochs, ~1,900 steps)

## 📓 Notebook Contents

1. **Environment Setup** - Dependencies and hardware configuration
2. **Dataset Exploration** - Visualization and distribution analysis
3. **Zero-Shot Baseline** - Evaluating pretrained model performance
4. **Fine-Tuning Experiments** - Three systematic LoRA configurations
5. **Checkpoint Evaluation** - Identifying best model per experiment
6. **Results Analysis** - Comprehensive quantitative comparison
7. **Production Deployment** - LoRA adapter merging for production use

## 🚀 Quick Start

### Path A: Engineering (Production API)

Deploy the system as a containerized service with built-in validation.

**Option 1: Docker (Recommended)**

```bash
docker build -t nutrition-detector .
docker run -p 8000:8000 nutrition-detector

```

**Option 2: Local Mock Mode (CPU-only)**

```bash
pip install -e .
uvicorn nutrition_detector.api.app:app --host 127.0.0.1 --port 8000 --loop asyncio
# Test it:
python scripts/test_api_local.py path/to/image.jpg

```

### Path B: Research (Training & Notebooks)

Reproduce the fine-tuning experiments and IoU benchmarks.

1. **Environment Setup**:
```bash
conda create -n vlm_research python=3.10 -y
conda activate vlm_research
pip install -e .

```


2. **Explore Notebooks**:
Open `notebooks/fine_tuning_qwen2_vl_for_object_detection_trl_A100_cleaned.ipynb` to view the full training pipeline and systematic LoRA comparisons.




**Note**: Training requires an A100 GPU (40GB VRAM). Evaluation can run on smaller GPUs with 4-bit quantization.

## 🚢 Production Deployment

The fine-tuned model is deployed on **NVIDIA Triton Inference Server** with vLLM backend for production-grade serving.

**Model**: [kulsoom-abdullah/qwen2-7b-nutrition-labels-detection](https://huggingface.co/kulsoom-abdullah/qwen2-7b-nutrition-labels-detection)

### Deployment Overview

**Production deployment** completed using two approaches:

1. **NVIDIA Triton Inference Server** - Enterprise-grade model serving
   - Model loaded: 15.53 GB (bfloat16)
   - Status: READY
   - Backend: vLLM 0.11.0 with Flash Attention
   - 📄 Setup: [TRITON_DEPLOYMENT.md](docs/TRITON_DEPLOYMENT.md)

2. **vLLM Standalone** - Quantization performance analysis
   - Baseline (bfloat16): 22.8 GB memory, ~600ms latency
   - FP8 quantized: 8.8 GB model weights (-45%), ~375ms latency (-37%)
   - Accuracy: Identical predictions (quantization preserves model performance)
   - 📊 Results: [QUANTIZATION_RESULTS.md](docs/QUANTIZATION_RESULTS.md)

**Key Finding**: FP8 quantization reduces model size by 45% and improves inference speed by 37% with zero accuracy loss, making it ideal for production deployment.

### Quick Inference Example

```python
import requests
from PIL import Image
import io
import base64

# Load and resize image
img = Image.open("nutrition_image.jpg")
if max(img.size) > 1024:
    ratio = 1024 / max(img.size)
    img = img.resize((int(img.size[0]*ratio), int(img.size[1]*ratio)))

# Convert to base64
buffered = io.BytesIO()
img.save(buffered, format="JPEG")
img_base64 = base64.b64encode(buffered.getvalue()).decode()

# Send request
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "kulsoom-abdullah/qwen2-7b-nutrition-labels-detection",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                {"type": "text", "text": "Detect the nutrition table. Provide bounding box in [x_min, y_min, x_max, y_max] format."}
            ]
        }],
        "max_tokens": 200,
        "temperature": 0.01
    }
)

print(response.json()['choices'][0]['message']['content'])
# Output: nutrition table(273,304,713,679)
```

## 📦 Repository Structure

```text
qwen2-vl-nutrition-detection/
├── src/nutrition_detector/      <-- Core production package (PEP 517)
│   ├── api/                     <-- FastAPI "Lobby" with Pydantic guardrails
│   ├── data/                    <-- Data collators & prompt masking logic
│   ├── model/                   <-- Model loading orchestration (LoRA/Quantization)
│   └── training/                <-- Managed SFTTrainer implementation
├── tests/                       <-- Unit tests & Mock objects for CPU verification
├── notebooks/                   <-- Archived training and research history
├── experiments/                 <-- Quantitative results and IoU visualizations
├── deploy/                      <-- Production Triton & vLLM configurations
├── docs/                        <-- Technical deep-dives (Quantization, Optimization)
├── scripts/                     <-- Utility scripts & API test clients
├── pyproject.toml               <-- Pinned dependencies for reproducibility
└── Dockerfile                   <-- Multi-stage production container

```

**Note**: Model checkpoints (900MB+) are excluded from the repository. The trained model is available on [HuggingFace Hub](https://huggingface.co/kulsoom-abdullah/qwen2-7b-nutrition-labels-detection).

## 🎓 Learning Resources

This project builds on concepts from:
- [Qwen2-VL Technical Report](https://qwenlm.github.io/blog/qwen2-vl/)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

## 📝 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- **Qwen Team** for the excellent Qwen2-VL model
- **OpenFoodFacts** for the nutrition table detection dataset
- **Daniel Voigt Godoy** for [A Hands-On Guide to Fine-Tuning Large Language Models](https://leanpub.com/finetuning)
- **RunPod** for accessible GPU infrastructure


