import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import vision_process

def get_model_and_processor(
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
    quantize: bool = True,
    use_lora: bool = True,
    dry_run: bool = False,
):
    """Initializes the Qwen2-VL model and processor with optimization configurations.

    Applies 4-bit NormalFloat (NF4) quantization and QLoRA adapters to allow fine-tuning
    a 7B parameter model on limited VRAM hardware (e.g., single A100 40GB).
    
    Also sets `vision_process.MAX_PIXELS` to restrict the resolution of input images,
    which is critical for preventing OOM errors caused by massive activation maps.

    Args:
        model_id: HuggingFace model identifier.
        quantize: If True, loads model with 4-bit NF4 quantization.
        use_lora: If True, applies LoRA adapters to linear layers.
        dry_run: If True, returns mock objects for testing without downloading weights.

    Returns:
        Tuple of (model, processor).
    """
    
    # Restricting max pixels caps the sequence length of visual tokens, directly reducing VRAM usage.
    vision_process.MAX_PIXELS = 600 * 28 * 28
    
    if dry_run:
        print(f"Mocking model loading for {model_id} (Dry Run)")
        # Dynamic import to keep production dependencies clean
        try:
            from tests.mocks import MockProcessor, MockModel
        except ImportError:
            raise ImportError("Dry run requires test dependencies. Ensure 'tests' package is accessible.")
            
        processor = MockProcessor()
        model = MockModel()
        return model, processor

    print(f"Loading {model_id}...")
    
    # Processor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # Quantization
    # NF4 provides better precision for normally distributed weights compared to standard 4-bit float.
    # Double quantization further reduces memory by quantizing the quantization constants.
    bnb_config = None
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Model
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA
    # Rank 8 is sufficient for this task; higher ranks increase memory without significant gains.
    # Targeting vision encoder attention blocks (attn.qkv) allows adapting visual features.
    if use_lora:
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj", 
                "v_proj",
                r"visual\.blocks\.\d+\.attn\.qkv"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, processor
