#!/usr/bin/env python3
"""
Merge LoRA adapters and prepare Qwen2-VL model for vLLM deployment.

This script:
1. Loads the base Qwen2-VL-7B model
2. Merges your fine-tuned LoRA adapters
3. Saves the merged model for vLLM/Triton deployment
4. Optionally pushes to HuggingFace Hub

Usage:
    python deploy_to_vllm.py --adapter_path qwen2-7b-nutrition-a100_exp1a/checkpoint-1626
"""

import argparse
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import os
from pathlib import Path


def merge_lora_adapters(
    base_model_id: str,
    adapter_path: str,
    output_dir: str,
    dtype: str = "bfloat16",
    push_to_hub: bool = False,
    hub_model_id: str = None,
):
    """
    Merge LoRA adapters with base model and save for deployment.

    Args:
        base_model_id: HuggingFace model ID (e.g., "Qwen/Qwen2-VL-7B-Instruct")
        adapter_path: Path to your LoRA checkpoint
        output_dir: Where to save the merged model
        dtype: Target precision (bfloat16, float16, or float32)
        push_to_hub: Whether to push to HuggingFace Hub
        hub_model_id: HuggingFace Hub repo name (e.g., "username/qwen2-7b-nutrition")
    """

    print("=" * 80)
    print("STEP 1: Loading base model...")
    print("=" * 80)

    # Map dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    # Load base model
    # Note: We load in the target dtype directly for efficiency
    print(f"Loading {base_model_id} with dtype={dtype}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"✓ Base model loaded successfully")
    print(f"  - Parameters: {model.num_parameters() / 1e9:.2f}B")
    print(f"  - Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

    print("\n" + "=" * 80)
    print("STEP 2: Loading LoRA adapters...")
    print("=" * 80)

    # Load LoRA adapters
    if not os.path.exists(adapter_path):
        raise ValueError(f"Adapter path not found: {adapter_path}")

    print(f"Loading adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    # Print trainable parameters info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ LoRA adapters loaded")
    print(f"  - Trainable params: {trainable_params / 1e6:.2f}M")
    print(f"  - All params: {total_params / 1e6:.2f}M")
    print(f"  - Trainable %: {100 * trainable_params / total_params:.4f}%")

    print("\n" + "=" * 80)
    print("STEP 3: Merging LoRA weights into base model...")
    print("=" * 80)

    # Merge and unload - this bakes the LoRA weights into the base model
    print("Merging LoRA adapters (this may take a few minutes)...")
    merged_model = model.merge_and_unload()

    print(f"✓ LoRA weights merged successfully")
    print(f"  - Final dtype: {merged_model.dtype}")

    print("\n" + "=" * 80)
    print("STEP 4: Saving merged model...")
    print("=" * 80)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save merged model
    print(f"Saving merged model to: {output_dir}")
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,  # Use safetensors format
        max_shard_size="5GB",  # Split into shards if needed
    )

    # Save processor (tokenizer + image processor)
    print("Saving processor...")
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    processor.save_pretrained(output_dir)

    print(f"✓ Model saved successfully")
    print(f"  - Location: {os.path.abspath(output_dir)}")
    print(f"  - Format: safetensors")

    # Optional: Push to HuggingFace Hub
    if push_to_hub:
        if not hub_model_id:
            raise ValueError("hub_model_id required when push_to_hub=True")

        print("\n" + "=" * 80)
        print("STEP 5: Pushing to HuggingFace Hub...")
        print("=" * 80)

        print(f"Pushing to: {hub_model_id}")
        print("Note: Make sure you're logged in with `huggingface-cli login`")

        merged_model.push_to_hub(
            hub_model_id,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        processor.push_to_hub(hub_model_id)

        print(f"✓ Model pushed to HuggingFace Hub")
        print(f"  - URL: https://huggingface.co/{hub_model_id}")

    print("\n" + "=" * 80)
    print("🎉 DEPLOYMENT PREPARATION COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Test locally with vLLM:")
    print(f"   vllm serve {output_dir if not push_to_hub else hub_model_id} --trust-remote-code")
    print("\n2. Deploy to Triton:")
    print("   - Create model repository structure")
    print("   - Add config.pbtxt and model.json")
    print("   - Launch Triton container")
    print("\nSee README for detailed deployment instructions.")

    return merged_model


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters and prepare for vLLM deployment"
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Base model ID from HuggingFace",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to LoRA checkpoint (e.g., qwen2-7b-nutrition-a100_exp1a/checkpoint-1626)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="qwen2-7b-nutrition-merged",
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Target precision for merged model",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push merged model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="HuggingFace Hub model ID (e.g., username/qwen2-7b-nutrition)",
    )

    args = parser.parse_args()

    # Validate adapter path
    if not os.path.exists(args.adapter_path):
        print(f"ERROR: Adapter path not found: {args.adapter_path}")
        print("\nAvailable checkpoints:")
        # Try to find checkpoints
        for exp_dir in ["qwen2-7b-nutrition-a100_exp1a", "qwen2-7b-nutrition-a100_exp1b", "qwen2-7b-nutrition-a100_exp2"]:
            if os.path.exists(exp_dir):
                checkpoints = [d for d in os.listdir(exp_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    print(f"\n{exp_dir}:")
                    for ckpt in sorted(checkpoints):
                        print(f"  - {exp_dir}/{ckpt}")
        return

    # Merge and save
    merge_lora_adapters(
        base_model_id=args.base_model_id,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        dtype=args.dtype,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )


if __name__ == "__main__":
    main()
