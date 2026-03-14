"""Validate submission file for BirdClef 2026."""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Validate submission for BirdClef 2026")
    parser.add_argument("--submission", type=str, required=True, help="Path to submission CSV")
    parser.add_argument("--sample_submission", type=str, default="data/birdclef-2026/sample_submission.csv", help="Path to sample submission")
    parser.add_argument("--taxonomy", type=str, default="data/birdclef-2026/taxonomy.csv", help="Path to taxonomy CSV")
    return parser.parse_args()


def validate_submission(submission_path: str, sample_path: str, taxonomy_path: str):
    """Validate submission file format and contents."""
    errors = []
    warnings = []
    
    print(f"Validating submission: {submission_path}")
    print("=" * 60)
    
    submission = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_path)
    taxonomy = pd.read_csv(taxonomy_path)
    
    print(f"Submission shape: {submission.shape}")
    print(f"Sample submission shape: {sample.shape}")
    
    if submission.shape != sample.shape:
        errors.append(f"Shape mismatch: submission {submission.shape} vs sample {sample.shape}")
    
    if list(submission.columns) != list(sample.columns):
        errors.append(f"Column mismatch: {list(submission.columns)} vs {list(sample.columns)}")
    
    row_id_col = 'row_id' if 'row_id' in submission.columns else submission.columns[0]
    label_cols = [c for c in submission.columns if c != row_id_col]
    
    if submission[row_id_col].isnull().any():
        errors.append("Found null values in row_id column")
    
    if not submission[row_id_col].equals(sample[row_id_col]):
        warnings.append("row_id order does not match sample submission")
    
    missing_species = set(sample.columns) - set(submission.columns)
    if missing_species:
        errors.append(f"Missing species columns: {missing_species}")
    
    extra_species = set(submission.columns) - set(sample.columns)
    if extra_species:
        warnings.append(f"Extra species columns (ignored): {extra_species}")
    
    probs = submission[label_cols].values
    
    if np.isnan(probs).any():
        errors.append("Found NaN values in probability matrix")
    
    if np.isinf(probs).any():
        errors.append("Found infinite values in probability matrix")
    
    if (probs < 0).any():
        errors.append(f"Found negative probabilities: min={probs.min()}")
    
    if (probs > 1).any():
        errors.append(f"Found probabilities > 1: max={probs.max()}")
    
    row_sums = probs.sum(axis=1)
    if (row_sums == 0).any():
        warnings.append(f"Found { (row_sums == 0).sum() } rows with all zero probabilities")
    
    expected_species = set(taxonomy['primary_label'].values)
    submitted_species = set(label_cols)
    
    missing_taxonomy = expected_species - submitted_species
    if missing_taxonomy:
        warnings.append(f"Missing {len(missing_taxonomy)} species from taxonomy")
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  ❌ {e}")
    
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠️  {w}")
    
    if not errors and not warnings:
        print("✅ Submission is valid!")
    elif not errors:
        print("\n✅ No errors found (warnings are non-critical)")
    else:
        print("\n❌ Submission has errors and cannot be used")
    
    print(f"\nProbability statistics:")
    print(f"  Min: {probs.min():.6f}")
    print(f"  Max: {probs.max():.6f}")
    print(f"  Mean: {probs.mean():.6f}")
    print(f"  Std: {probs.std():.6f}")
    
    print(f"\nRows with predictions above threshold:")
    for thresh in [0.1, 0.25, 0.5]:
        count = (probs.max(axis=1) > thresh).sum()
        print(f"  > {thresh}: {count} ({100*count/len(probs):.1f}%)")
    
    return len(errors) == 0


def main():
    args = parse_args()
    
    submission_path = Path(args.submission)
    sample_path = Path(args.sample_submission)
    taxonomy_path = Path(args.taxonomy)
    
    if not submission_path.exists():
        print(f"Error: Submission file not found: {submission_path}")
        return 1
    
    if not sample_path.exists():
        print(f"Warning: Sample submission not found: {sample_path}")
        sample_path = None
    
    if not taxonomy_path.exists():
        print(f"Warning: Taxonomy not found: {taxonomy_path}")
        taxonomy_path = None
    
    is_valid = validate_submission(
        str(submission_path),
        str(sample_path) if sample_path else None,
        str(taxonomy_path) if taxonomy_path else None,
    )
    
    return 0 if is_valid else 1


if __name__ == "__main__":
    exit(main())
