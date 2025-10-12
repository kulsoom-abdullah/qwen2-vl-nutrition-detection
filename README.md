# Fine-Tuning Qwen2-VL-7B for Nutrition Table Detection

A systematic exploration of parameter-efficient fine-tuning strategies for vision-language models on object detection tasks.

## 🎯 Project Overview

This project fine-tunes **Qwen2-VL-7B** to detect nutrition tables in product images using **QLoRA (4-bit quantized LoRA)** on RunPod A100 GPUs. The notebook documents three experiments comparing different LoRA configurations and training strategies.

![Prediction Example](images/failure_analysis_exp1a.png)

**Dataset**: [OpenFoodFacts Nutrition Table Detection](https://huggingface.co/datasets/openfoodfacts/nutrition-table-detection) (1,106 training images, 123 test images)

## 📊 Results Summary

| Experiment | Mean IoU | F1@0.5 | Improvement |
|:-----------|:--------:|:------:|:-----------:|
| **Baseline (Zero-Shot)** | 0.590 | 0.654 | - |
| **Exp 1a: LLM LoRA + Masking** ⭐ | **0.771** | **0.893** | **+30.7%** |
| **Exp 1b: LLM LoRA (No Masking)** | 0.745 | 0.870 | +26.3% |
| **Exp 2: Vision+LLM LoRA + Masking** | 0.748 | 0.863 | +26.8% |

## 🔑 Key Findings

1. **LLM-only LoRA is optimal** - Tuning vision encoder provides minimal benefit
2. **Prompt masking improves performance** - Computing loss only on model responses reduces noise
3. **QLoRA enables efficient fine-tuning** - 4-bit quantization with LoRA (rank=64) achieves strong results on a single A100
4. **Consistent evaluation is critical** - Using identical setups (quantization, attention, resolution) across all experiments ensures fair comparison

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

```bash
# Clone repository
git clone <repo-url>
cd transformers

# Install dependencies
pip install -q transformers[torch] trl peft datasets accelerate bitsandbytes qwen-vl-utils

# Open notebook
jupyter notebook fine_tuning_qwen2_vl_for_object_detection_trl_A100_backup.ipynb
```

**Note**: Training requires an A100 GPU (40GB VRAM). Evaluation can run on smaller GPUs with 4-bit quantization.

## 🚢 Production Deployment

The fine-tuned model is deployed via vLLM with OpenAI-compatible API and available on HuggingFace Hub:
- **Model**: [kulsoom-abdullah/qwen2-7b-nutrition-labels-detection](https://huggingface.co/kulsoom-abdullah/qwen2-7b-nutrition-labels-detection)

### vLLM Deployment

```bash
# Serve with vLLM (OpenAI-compatible API)
vllm serve kulsoom-abdullah/qwen2-7b-nutrition-labels-detection \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --port 8000
```

### Triton Inference Server (Optional)

For production deployments requiring advanced features, the model can be served via NVIDIA Triton with vLLM backend.

**Requirements:**
- NVIDIA Triton Docker image: `nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3`
- transformers >= 4.45.0
- vLLM >= 0.6.0
- GPU: 40GB+ VRAM (A100 or equivalent)

**Setup:**

```bash
# Upgrade packages in container
pip install --upgrade "transformers>=4.45.0" "vllm>=0.6.0" qwen-vl-utils

# Create model repository structure
mkdir -p /workspace/model_repository/qwen2_nutrition/1

# Create config (see triton_config.pbtxt in repo)
# Start Triton
tritonserver --model-repository=/workspace/model_repository
```

See `triton_config.pbtxt` for the complete Triton configuration.

### Inference Example

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

```
transformers/
├── fine_tuning_qwen2_vl_for_object_detection_trl_A100_cleaned.ipynb  # Cleaned notebook (no outputs, 100KB)
├── fine_tuning_qwen2_vl_A100_with_outputs.html                       # HTML export with all outputs rendered (21MB)
├── qwen2-7b-nutrition-baseline/                                      # Baseline results
├── qwen2-7b-nutrition-a100_exp1a/                                    # Exp 1a results (CSVs, PNGs)
├── qwen2-7b-nutrition-a100_exp1b/                                    # Exp 1b results (CSVs, PNGs)
├── qwen2-7b-nutrition-a100_exp2/                                     # Exp 2 results (CSVs, PNGs)
├── images/                                                           # Visualization outputs
├── deploy_to_vllm.py                                                 # LoRA merge script
├── triton_config.pbtxt                                               # Triton vLLM backend config
└── README.md
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
