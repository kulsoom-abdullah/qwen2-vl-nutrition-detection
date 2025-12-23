# Memory Optimization Strategy - Qwen2-VL Fine-Tuning

## Overview
This document details the memory optimization techniques used to fine-tune Qwen2-VL-7B (7 billion parameters) on a single A100 80GB GPU for nutrition label detection.

## Training Configuration

**Model**: Qwen2-VL-7B-Instruct (~7.6B parameters)  
**Task**: Object detection (nutrition label bounding boxes)  
**Hardware**: Single NVIDIA A100 80GB GPU  
**Training samples**: ~1,100 images  
**Context**: Vision-language model with both vision encoder and language decoder

**Note**: Despite the "7B" naming, this model actually has approximately 7.6-7.8B parameters (HuggingFace lists as "8B params").

## Memory Optimization Hierarchy

The techniques below are ordered by **impact magnitude** - from largest memory savings to smallest. This ordering is critical for efficient resource utilization.

### 1. Quantization (Largest Impact: ~75% reduction)

**Implementation**: 4-bit NF4 quantization via bitsandbytes

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
```

**Memory Impact**:
- Base model FP16: 7B × 2 bytes = 14 GB
- Base model 4-bit NF4: 7B × 0.5 bytes = 3.5 GB
- **Savings: 10.5 GB (75% reduction)**

**Quality Impact**: Minimal - 4-bit NF4 is specifically designed to preserve model quality

### 2. LoRA Adapters (Parameter Efficiency: ~99% reduction in trainable params)

**Implementation**: Low-Rank Adaptation with rank=8, alpha=16

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                           # Rank (reduced from typical 16-64)
    lora_alpha=16,                 # Scaling factor
    target_modules=[
        "q_proj",                  # Query projection only
        "v_proj",                  # Value projection only
        # Note: Excluded k_proj, o_proj, gate_proj, up_proj, down_proj
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

**Design Rationale**:
- **Query and Value only**: Research shows q_proj and v_proj capture most of the adaptation capacity
- **Lower rank (r=8)**: Sufficient for this task while minimizing parameters
- **Selective targeting**: Fewer modules = fewer parameters = less memory

**Parameter Impact**:
- Full model parameters: ~7,600,000,000
- LoRA trainable parameters: ~2,100,000 (calculated: 32 layers × 2 modules × 4096 × 8)
- **Trainable %: ~0.03%**

**Memory Impact**:
- LoRA parameters: ~8 MB (2.1M × 4 bytes FP32)
- Gradients: ~4 MB (2.1M × 2 bytes BF16)
- Optimizer states: ~25 MB (2.1M × 12 bytes for AdamW)
- **Total LoRA memory: ~37 MB**
- **Comparison**: Full 7.6B training would need ~91 GB for parameters + gradients + optimizer states

### 3. Attention Mechanism (SDPA vs Flash Attention)

**Implementation**: PyTorch's Scaled Dot Product Attention (SDPA)

```python
# For stable inference - disable Flash Attention optimizations
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)  # Use standard math implementation
```

**Why SDPA instead of Flash Attention 2?**
- **Compatibility**: More reliable with 4-bit quantization + BFloat16
- **Stability**: No kernel fallback issues during inference
- **Consistency**: Same attention mechanism across all evaluations
- **Sufficient Performance**: Evaluation not bottlenecked by attention (model loading takes longer)

**Memory Impact**:
- SDPA vs standard attention: Modest memory savings (~2-3 GB)
- Flash Attention 2 would save more (~7-8 GB) but compatibility issues with quantization
- **Trade-off accepted**: Stability over maximum optimization

**Training vs Evaluation**:
- **Training**: Used default attention (whatever PyTorch selected) for memory efficiency
- **Evaluation**: Explicitly specified SDPA for consistent, stable inference across all checkpoints

### 4. Gradient Accumulation + Checkpointing

**Implementation**: Accumulate gradients + checkpoint activations

```python
from trl import SFTConfig

sft_config = SFTConfig(
    per_device_train_batch_size=1,          # Micro-batch size
    gradient_accumulation_steps=4,          # Accumulate over 4 steps
    gradient_checkpointing=True,            # Recompute activations in backward pass
    # Effective batch size = 1 × 4 = 4
)
```

**Memory Impact**:
- **Gradient accumulation**: Allows effective batch size of 4 while only storing activations for batch size 1
- **Gradient checkpointing**: Trades ~40% extra compute for ~50% activation memory savings
- **Combined savings**: ~60-70% of activation memory vs batch_size=4 without checkpointing

**Trade-offs**:
- Slightly slower training (~40% overhead from recomputation)
- Essential for fitting larger batches on single GPU
- Worth it when memory-constrained

### 5. Mixed Precision Training (Standard Practice)

**Implementation**: BFloat16 automatic mixed precision

```python
training_args = TrainingArguments(
    bf16=True,  # Use BF16 on A100
    bf16_full_eval=True,
)
```

**Memory Impact**:
- Activations: FP32 (4 bytes) → BF16 (2 bytes) = 50% reduction
- **Savings: ~3-5 GB depending on batch size**

**Why BF16 over FP16**:
- Better numerical range (same exponent bits as FP32)
- No loss scaling required
- More stable training

### 6. 8-bit Optimizer (Minimal Impact for LoRA)

**Implementation**: Paged AdamW 8-bit via TRL/bitsandbytes

```python
from trl import SFTConfig

sft_config = SFTConfig(
    optim="paged_adamw_8bit",  # 8-bit optimizer with paging
    learning_rate=1e-4,
    weight_decay=0.01,
)
```

**Memory Impact for LoRA**:
- Standard AdamW: 2.1M params × 12 bytes = 25 MB
- 8-bit Paged AdamW: 2.1M params × 6 bytes = 13 MB
- **Savings: 12 MB**

**Conclusion**: Almost negligible savings (~12 MB) when using selective LoRA with only 2.1M trainable parameters. The optimizer state memory is dwarfed by the base model (3.5 GB) and activation memory (several GB).

**Quality Impact**: None - 8-bit optimizer produces equivalent results to standard AdamW

**Verdict**: Not necessary for LoRA training, but doesn't hurt either. More impactful for full fine-tuning.

## Memory Budget Breakdown

**Total GPU Memory**: 80 GB

**Usage Distribution**:
```
Base model (4-bit):              3.5 GB   ( 4%)
LoRA parameters:                <0.1 GB   (<1%)
Optimizer states (8-bit):       <0.1 GB   (<1%)
Gradients:                      <0.1 GB   (<1%)
Activations (batch=1):           8.0 GB   (10%)
Gradient checkpointing:         -4.0 GB   (saved)
SDPA efficiency:                -2.0 GB   (saved vs naive attention)
KV Cache:                        2.0 GB   ( 3%)
System overhead:                 1.0 GB   ( 1%)
----------------------------------------
Peak Usage:                     ~9 GB   (11%)
Headroom:                      ~71 GB   (89%)
```

**Note**: These are estimated values based on model architecture and configuration. Actual memory usage may vary during training.

## Key Insights

### What Mattered Most

1. **Quantization (4-bit)**: Primary enabler - without this, model wouldn't fit
2. **LoRA**: Made training feasible with minimal quality loss
3. **Flash Attention 2**: Critical for batch processing and longer sequences
4. **Gradient Accumulation**: Allowed optimal batch sizes

### What Didn't Matter Much

1. **8-bit Optimizer**: Only saved 300 MB when training 50M LoRA parameters
   - Would matter more for full fine-tuning (7B parameters)
   - Not worth the added complexity for LoRA setups

2. **Aggressive Micro-batch Sizes**: With QLoRA + Flash Attention, we had plenty of memory
   - Could use batch_size=2 or even 4 comfortably
   - No need to go down to batch_size=1

## Training Performance Impact

**Training Configuration**:
- Epochs: 7
- Effective batch size: 4 (1 per device × 4 accumulation steps)
- Learning rate: 1e-4 with cosine schedule
- Warmup: 3% of training steps
- Total steps: ~1,900 per experiment

**Performance Notes**:
- Training time: ~2 hours per experiment on A100 80GB
- See [README.md](README.md) for detailed accuracy metrics and experimental results

**Quality Impact from Optimizations**:
- No observable degradation from 4-bit quantization
- LoRA with selective modules (q_proj, v_proj) maintained task performance
- SDPA provided stable inference without accuracy loss
- Achieved 30.7% IoU improvement over baseline (0.590 → 0.771)

## Recommendations for Future Projects

### Always Use (High Impact):
1. 4-bit quantization for large models (>1B params)
2. LoRA with selective target modules (q_proj, v_proj often sufficient)
3. SDPA or Flash Attention based on compatibility needs
4. Mixed precision (BF16 on A100/H100, FP16 elsewhere)

### Use If Needed (Medium Impact):
1. Gradient accumulation if memory-constrained
2. Dynamic sequence length batching
3. Gradient checkpointing for very long sequences

### Skip for LoRA Training (Low Impact):
1. 8-bit optimizers - only saves ~12 MB with 2.1M LoRA parameters
2. Aggressive gradient checkpointing - balance memory vs speed
3. Model parallelism - unnecessary for 7B models on A100

## Cost-Benefit Analysis

| Technique | Implementation Complexity | Memory Saved | Quality Impact | Worth It? |
|-----------|--------------------------|--------------|----------------|-----------|
| 4-bit Quantization | Low | +++++ (10+ GB) | None | ✓ Always |
| LoRA | Low | +++++ (55+ GB) | Minimal | ✓ Always |
| Flash Attention | Trivial | ++++ (7-8 GB) | None | ✓ Always |
| Gradient Accumulation | Trivial | +++ (varies) | None | ✓ If needed |
| Mixed Precision (BF16) | Trivial | ++ (3-5 GB) | None | ✓ Always |
| 8-bit Optimizer | Low | + (300 MB) | None | ✗ Not worth it |

## References

- QLoRA Paper: https://arxiv.org/abs/2305.14314
- Flash Attention 2: https://arxiv.org/abs/2307.08691
- bitsandbytes: https://github.com/TimDettmers/bitsandbytes
- PEFT Library: https://github.com/huggingface/peft

## Appendix: Commands Used

### Training Command
```bash
python train_qwen2vl.py \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --data_path nutrition_5k_qa.json \
    --output_dir qwen2-7b-nutrition-a100_exp1a \
    --num_train_epochs 7 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --bf16 True \
    --optim paged_adamw_8bit \
    --use_lora True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_modules q_proj v_proj
```

### Memory Monitoring
```bash
# Watch GPU memory usage during training
watch -n 1 nvidia-smi
```

---

**Last Updated**: November 2025  
**Experiment**: Qwen2-VL Nutrition Label Detection  
**Author**: Kulsoom Abdullah