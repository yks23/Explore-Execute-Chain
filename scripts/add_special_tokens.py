#!/usr/bin/env python3
"""
E2C Special Token Addition Script

This script adds the required special tokens to a base model for E2C training.
Must be run before training if using a locally downloaded base model.

Required tokens:
  - <EXPLORATION>  : Marks the start of exploration phase
  - </EXPLORATION> : Marks the end of exploration phase
  - <EXECUTION>    : Marks the start of execution phase
  - </EXECUTION>   : Marks the end of execution phase

Usage:
    python scripts/add_special_tokens.py --model_path /path/to/model --output_path /path/to/output
"""

import argparse
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def add_special_tokens(model_path, output_path=None, dry_run=False):
    """
    Add E2C special tokens to a model
    
    Args:
        model_path: Path to the base model
        output_path: Path to save the updated model (default: model_path + "_e2c")
        dry_run: If True, only show what would be done without modifying files
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    
    if output_path is None:
        output_path = Path(str(model_path) + "_e2c")
    else:
        output_path = Path(output_path)
    
    print("=" * 70)
    print("E2C Special Token Addition")
    print("=" * 70)
    print(f"Input model:  {model_path}")
    print(f"Output model: {output_path}")
    print(f"Dry run:      {dry_run}")
    print("=" * 70)
    print()
    
    # Define special tokens
    special_tokens = [
        "<EXPLORATION>",
        "</EXPLORATION>",
        "<EXECUTION>",
        "</EXECUTION>"
    ]
    
    print("Special tokens to add:")
    for i, token in enumerate(special_tokens, 1):
        print(f"  {i}. {token}")
    print()
    
    if dry_run:
        print("DRY RUN MODE - No changes will be made")
        print(f"Would load model from: {model_path}")
        print(f"Would add {len(special_tokens)} special tokens")
        print(f"Would save updated model to: {output_path}")
        return
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True
    )
    original_vocab_size = len(tokenizer)
    print(f"  Original vocabulary size: {original_vocab_size}")
    
    # Check if tokens already exist
    existing_tokens = []
    for token in special_tokens:
        if token in tokenizer.get_vocab():
            existing_tokens.append(token)
    
    if existing_tokens:
        print(f"\n⚠️  Warning: {len(existing_tokens)} token(s) already exist in vocabulary:")
        for token in existing_tokens:
            print(f"    - {token}")
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted by user.")
            return
    
    # Add special tokens
    print(f"\nAdding {len(special_tokens)} special tokens...")
    num_added_tokens = tokenizer.add_special_tokens({
        "additional_special_tokens": special_tokens
    })
    new_vocab_size = len(tokenizer)
    print(f"  Added tokens: {num_added_tokens}")
    print(f"  New vocabulary size: {new_vocab_size}")
    
    if num_added_tokens == 0:
        print("\n✓ All tokens already present in tokenizer. No changes needed.")
    else:
        print(f"  Vocabulary increased by: {new_vocab_size - original_vocab_size} tokens")
    
    # Load model
    print("\nLoading model...")
    print("  (This may take a few minutes depending on model size)")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu"  # Load to CPU to avoid GPU memory issues
    )
    
    original_embedding_size = model.get_input_embeddings().weight.shape[0]
    print(f"  Original embedding size: {original_embedding_size}")
    
    # Resize model embeddings
    print("\nResizing model embedding table...")
    model.resize_token_embeddings(len(tokenizer))
    new_embedding_size = model.get_input_embeddings().weight.shape[0]
    print(f"  New embedding size: {new_embedding_size}")
    print(f"  Embeddings increased by: {new_embedding_size - original_embedding_size}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save updated model
    print(f"\nSaving updated model to: {output_path}")
    model.save_pretrained(str(output_path))
    print("  ✓ Model saved")
    
    # Save updated tokenizer
    print(f"Saving updated tokenizer to: {output_path}")
    tokenizer.save_pretrained(str(output_path))
    print("  ✓ Tokenizer saved")
    
    print("\n" + "=" * 70)
    print("✓ Successfully added E2C special tokens!")
    print("=" * 70)
    print(f"\nUpdated model location: {output_path}")
    print("\nNext steps:")
    print(f"  1. Use the updated model for training:")
    print(f"     export MODEL_PATH=\"{output_path}\"")
    print(f"     bash scripts/e2c_sft.sh")
    print("\n  2. Or modify the training script to use this path directly")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Add E2C special tokens to a base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add tokens to a model
  python scripts/add_special_tokens.py --model_path /path/to/qwen3-8b

  # Specify custom output path
  python scripts/add_special_tokens.py \\
    --model_path /path/to/qwen3-8b \\
    --output_path /path/to/qwen3-8b-e2c

  # Dry run to see what would be done
  python scripts/add_special_tokens.py \\
    --model_path /path/to/qwen3-8b \\
    --dry_run

Special tokens that will be added:
  <EXPLORATION>  - Start of exploration phase
  </EXPLORATION> - End of exploration phase
  <EXECUTION>    - Start of execution phase
  </EXECUTION>   - End of execution phase
        """
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the base model directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the updated model (default: model_path + '_e2c')"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    try:
        add_special_tokens(
            model_path=args.model_path,
            output_path=args.output_path,
            dry_run=args.dry_run
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

