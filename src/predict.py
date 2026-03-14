"""Prediction script for BirdClef 2026."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Predict with BirdClef 2026 model")
    parser.add_argument("--input", type=str, required=True, help="Input audio file or directory")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Model checkpoint path")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output submission file")
    args = parser.parse_args()

    print(f"Predicting:")
    print(f"  Input: {args.input}")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")

    # TODO: Implement prediction logic


if __name__ == "__main__":
    main()
