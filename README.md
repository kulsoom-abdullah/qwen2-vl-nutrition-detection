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

The fine-tuned model is deployed on **NVIDIA Triton Inference Server** with vLLM backend for production-grade serving.

**Model**: [kulsoom-abdullah/qwen2-7b-nutrition-labels-detection](https://huggingface.co/kulsoom-abdullah/qwen2-7b-nutrition-labels-detection)

### Deployment Overview

**Production deployment** completed using two approaches:

1. **NVIDIA Triton Inference Server** - Enterprise-grade model serving
   - Model loaded: 15.53 GB (bfloat16)
   - Status: READY
   - Backend: vLLM 0.11.0 with Flash Attention
   - 📄 Setup: [TRITON_DEPLOYMENT.md](TRITON_DEPLOYMENT.md)

2. **vLLM Standalone** - Quantization performance analysis
   - Baseline (bfloat16): 22.8 GB memory, ~600ms latency
   - FP8 quantized: 8.8 GB model weights (-45%), ~375ms latency (-37%)
   - Accuracy: Identical predictions (quantization preserves model performance)
   - 📊 Results: [QUANTIZATION_RESULTS.md](QUANTIZATION_RESULTS.md)

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

```
transformers/
├── fine_tuning_qwen2_vl_for_object_detection_trl_A100_cleaned.ipynb  # Cleaned training notebook
├── fine_tuning_qwen2_vl_A100_with_outputs.html                       # Full notebook with outputs (21MB)
├── qwen2-7b-nutrition-baseline/                                      # Zero-shot baseline results
├── qwen2-7b-nutrition-a100_exp1a/                                    # Exp 1a: LLM LoRA + masking ⭐
├── qwen2-7b-nutrition-a100_exp1b/                                    # Exp 1b: LLM LoRA (no masking)
├── qwen2-7b-nutrition-a100_exp2/                                     # Exp 2: Vision+LLM LoRA + masking
├── images/                                                           # Visualization outputs
├── deploy_to_vllm.py                                                 # LoRA adapter merge script
├── TRITON_DEPLOYMENT.md                                              # Triton setup & quantization benchmark
├── memory-optimization-doc.md                                        # Memory optimization strategies
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
