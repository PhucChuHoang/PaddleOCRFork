#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to load an exported Paddle model via paddle.jit.load, count parameters,
and display a layer-wise summary using paddle.summary.

Usage examples:
  python simple_model_summary.py ./inference/ch_PP-OCRv3_rec_infer
  python simple_model_summary.py ./inference/ch_PP-OCRv3_rec_infer --input_size 3 48 320
"""

import sys
import argparse
import paddle


def count_parameters(model):
    """Count total parameters in the model."""
    total_params = 0
    try:
        for param in model.parameters():
            total_params += int(param.numel())
    except Exception:
        # If model.parameters() is not available or model is not a Layer-like object
        total_params = None
    return total_params


def format_number(num):
    """Format large numbers with appropriate units."""
    if num is None:
        return "N/A"
    num = float(num)
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return str(int(num))


def try_summary(model, input_size):
    try:
        # Try with batch size = 1
        paddle.summary(model, input_size=(1, *input_size))
        return True
    except Exception as e:
        print(f"  • paddle.summary failed for input_size=(1, {input_size}): {e}")
    # If your Paddle version supports batch_size param, you could also try:
    try:
        paddle.summary(model, input_size=input_size, batch_size=1)
        return True
    except Exception:
        pass
    return False



def main():
    parser = argparse.ArgumentParser(
        description="Load a Paddle model (exported with paddle.jit.save) and show parameter count and summary."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the exported model directory or file to load with paddle.jit.load"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=3,
        metavar=('C', 'H', 'W'),
        help=(
            "Explicit input size (channels, height, width) for paddle.summary. "
            "E.g. --input_size 3 48 320"
        ),
    )
    args = parser.parse_args()

    model_path = args.model_path
    print(f"Loading model from: {model_path}")

    try:
        model = paddle.jit.load(model_path)
        model.eval()  # set to eval mode if applicable
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)

    # Count parameters
    total_params = count_parameters(model)
    formatted = format_number(total_params) if total_params is not None else "N/A"
    if total_params is not None:
        print(f"\nModel Parameter Summary:")
        print(f"  Total parameters: {formatted} ({total_params:,})")
    else:
        print("\nCould not count parameters via model.parameters().")

    # Prepare a list of input sizes to try for summary
    input_sizes_to_try = []
    if args.input_size:
        input_sizes_to_try.append(tuple(args.input_size))
    else:
        # Common shapes for PaddleOCR exported models; adjust or extend if needed
        input_sizes_to_try.extend([
            (3, 48, 320),   # recognition
            (3, 48, 192),   # small recog
            (3, 640, 640),  # detection
            (3, 224, 224),  # generic
        ])

    print("\nAttempting detailed summary via paddle.summary:")
    summary_shown = False
    for inp in input_sizes_to_try:
        print(f"Trying input_size={inp} ...")
        success = try_summary(model, input_size=inp)
        if success:
            summary_shown = True
            break

    if not summary_shown:
        print("\nCould not generate a detailed summary with the tried input sizes.")
        print("You may need to specify a different --input_size or ensure the model architecture supports paddle.summary.")
        print("Example: python simple_model_summary.py <model_path> --input_size 3 48 320")


if __name__ == "__main__":
    main()
