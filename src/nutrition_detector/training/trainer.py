import argparse
import os
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from torchvision.ops import box_iou
from unittest.mock import MagicMock

from nutrition_detector.data.dataset import (
    create_chat_format, 
    VLMDataCollator, 
    parse_bounding_boxes
)
from nutrition_detector.model.loader import get_model_and_processor

def compute_metrics(eval_pred, processor) -> dict:
    """Calculates Mean IoU and F1-score for object detection tasks.

    Unlike standard accuracy, this computes geometric overlap (IoU) between predicted
    and ground-truth bounding boxes. Uses a greedy matching algorithm to handle
    multiple detections.

    Args:
        eval_pred: Tuple of (predictions, labels) from the trainer.
        processor: The model processor used for decoding text tokens.

    Returns:
        Dictionary containing 'mean_gt_iou', 'precision', 'recall', and 'f1'.
    """
    predictions, labels = eval_pred
    
    # Handle tuple if passed
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Decode predictions
    decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 with pad token id in a copy of labels, then decode
    labels_copy = labels.copy()
    labels_copy[labels_copy == -100] = processor.tokenizer.pad_token_id
    decoded_labels = processor.batch_decode(labels_copy, skip_special_tokens=True)

    total_iou = 0.0
    tp = fp = fn = 0
    total_gt = 0
    iou_threshold = 0.5

    for pred_text, label_text in zip(decoded_preds, decoded_labels):
        pred_boxes = parse_bounding_boxes(pred_text)  # [x_min, y_min, x_max, y_max]
        gt_boxes = parse_bounding_boxes(label_text)   # [x_min, y_min, x_max, y_max] from assistant string
        
        if not gt_boxes and not pred_boxes:
            continue
        if not pred_boxes:
            fn += len(gt_boxes)
            total_gt += len(gt_boxes)
            continue
        if not gt_boxes:
            fp += len(pred_boxes)
            continue

        pred_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
        gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32)

        iou_matrix = box_iou(pred_tensor, gt_tensor)
        if iou_matrix.numel() == 0:
            fn += len(gt_boxes)
            fp += len(pred_boxes)
            total_gt += len(gt_boxes)
            continue

        # greedy match
        all_pairs = [
            (iou_matrix[p, g].item(), p, g)
            for p in range(iou_matrix.shape[0])
            for g in range(iou_matrix.shape[1])
        ]
        all_pairs.sort(reverse=True)

        matched_preds = set()
        matched_gts = set()
        matched_iou_sum = 0.0
        for iou, p, g in all_pairs:
            if iou < iou_threshold:
                break
            if p in matched_preds or g in matched_gts:
                continue
            matched_preds.add(p)
            matched_gts.add(g)
            matched_iou_sum += iou

        tp += len(matched_preds)
        fp += len(pred_boxes) - len(matched_preds)
        fn += len(gt_boxes) - len(matched_preds)

        total_iou += matched_iou_sum
        total_gt += len(gt_boxes)

    mean_iou = total_iou / total_gt if total_gt else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "mean_gt_iou": mean_iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def train(args):
    """Executes the QLoRA fine-tuning pipeline.

    Orchestrates data loading, model initialization, and the training loop.
    Uses gradient accumulation to simulate larger batch sizes on constrained hardware.

    Args:
        args: Namespace containing CLI arguments.
    """
    # 1. Load Model & Processor
    model, processor = get_model_and_processor(
        model_id=args.model_id,
        dry_run=args.dry_run
    )

    # 2. Load Dataset
    if args.dry_run:
        print("Creating mock dataset for dry run...")
        # Create a dummy sample that matches the expected structure
        dummy_image = MagicMock()
        dummy_image.copy.return_value = dummy_image
        dummy_image.size = (100, 100)
        
        # Mock dataset with enough samples
        mock_sample = {
            "image": dummy_image,
            "objects": {
                "bbox": [[0.1, 0.1, 0.2, 0.2]],
                "category_name": ["nutrition-table"]
            }
        }
        train_dataset = [create_chat_format(mock_sample, downsize=False) for _ in range(2)]
        eval_dataset = [create_chat_format(mock_sample, downsize=False) for _ in range(2)]
    else:
        print(f"Loading dataset {args.dataset_id}...")
        ds_train = load_dataset(args.dataset_id, split="train")
        ds_val = load_dataset(args.dataset_id, split="val")
        
        if args.max_samples:
            ds_train = ds_train.select(range(args.max_samples))
            ds_val = ds_val.select(range(args.max_samples))

        print("Formatting datasets...")
        train_dataset = [create_chat_format(sample) for sample in ds_train]
        eval_dataset = [create_chat_format(sample) for sample in ds_val]

    # 3. Data Collator
    data_collator = VLMDataCollator(processor=processor)

    # 4. Config
    # Gradient accumulation of 4 steps allows effective batch size of 4 (1 * 4) 
    # to stabilize training gradients without exceeding VRAM limits.
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=not args.dry_run, # Disable in dry run if using mocks
        bf16=not args.dry_run, # Mocks might not support bf16
        tf32=not args.dry_run,
        learning_rate=1e-4,
        logging_steps=10,
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        save_strategy="epoch",
        eval_strategy="no" if args.dry_run else "epoch", # Skip eval loop in dry run to keep it simple, or mock it
        use_cpu=args.dry_run, # Force CPU for dry run
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, processor),
        processing_class=processor,
    )

    # 6. Train
    print("Starting training...")
    if args.dry_run:
        # Just mock the training step
        print("Dry run: Training simulated.")
    else:
        trainer.train()
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-VL for Nutrition Detection")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--dataset_id", type=str, default="openfoodfacts/nutrition-table-detection")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, help="Limit number of samples for testing")
    parser.add_argument("--dry_run", action="store_true", help="Run without GPU/heavy loading for testing")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
