"""
DNAfold2 Command Line Interface

Provides a command-line interface for DNA structure folding.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import FoldingConfig, validate_sequence
from .core import DNAFolder, fold


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="dnafold2",
        description="DNAfold2: Ab initio DNA 3D structure prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fold a sequence directly
  dnafold2 fold --sequence ATCCTAGTTATAGGAT --output results/

  # Fold from a sequence file
  dnafold2 fold --input seq.dat --config config.dat --output results/

  # Generate a configuration file
  dnafold2 config --output my_config.dat

  # Show version
  dnafold2 --version
"""
    )
    
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version and exit"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Fold command
    fold_parser = subparsers.add_parser(
        "fold",
        help="Fold a DNA sequence to predict 3D structure"
    )
    
    input_group = fold_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--sequence", "-s",
        type=str,
        help="DNA sequence (e.g., ATCCTAGTTATAGGAT)"
    )
    input_group.add_argument(
        "--input", "-i",
        type=Path,
        help="Path to sequence file (seq.dat)"
    )
    
    fold_parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to configuration file (config.dat)"
    )
    fold_parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results"),
        help="Output directory (default: results/)"
    )
    fold_parser.add_argument(
        "--method", "-m",
        choices=["remc", "sa"],
        default="remc",
        help="Sampling method: remc (Replica Exchange MC) or sa (Simulated Annealing)"
    )
    fold_parser.add_argument(
        "--steps",
        type=int,
        help="Number of folding steps"
    )
    fold_parser.add_argument(
        "--na-concentration",
        type=float,
        help="Na+ concentration in mM"
    )
    fold_parser.add_argument(
        "--mg-concentration",
        type=float,
        help="Mg2+ concentration in mM"
    )
    fold_parser.add_argument(
        "--n-structures",
        type=int,
        help="Number of structures to predict"
    )
    fold_parser.add_argument(
        "--n-threads",
        type=int,
        help="Number of temperature replicas (1-20, should match available CPUs for REMC)"
    )
    fold_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information"
    )
    
    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Generate a configuration file"
    )
    config_parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("config.dat"),
        help="Output path for configuration file"
    )
    config_parser.add_argument(
        "--method",
        choices=["remc", "sa"],
        default="remc",
        help="Sampling method"
    )
    config_parser.add_argument(
        "--steps",
        type=int,
        default=500000,
        help="Number of folding steps"
    )
    config_parser.add_argument(
        "--optimize-steps",
        type=int,
        default=100000,
        help="Number of optimization steps"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a DNA sequence"
    )
    validate_parser.add_argument(
        "sequence",
        type=str,
        help="DNA sequence to validate"
    )
    
    # Info command
    subparsers.add_parser(
        "info",
        help="Show installation information"
    )
    
    return parser


def cmd_fold(args: argparse.Namespace) -> int:
    """Execute the fold command."""
    # Get sequence
    if args.sequence:
        sequence = args.sequence
    else:
        if not args.input.exists():
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            return 1
        sequence = args.input.read_text().strip()
    
    # Validate sequence
    try:
        sequence = validate_sequence(sequence)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Build configuration
    if args.config and args.config.exists():
        config = FoldingConfig.from_file(args.config)
    else:
        config = FoldingConfig()
    
    # Override with command-line arguments
    if args.method:
        config.sampling_method = args.method
    if args.steps:
        config.folding_steps = args.steps
    if args.na_concentration is not None:
        config.na_concentration = args.na_concentration
    if args.mg_concentration is not None:
        config.mg_concentration = args.mg_concentration
    if args.n_structures:
        config.n_structures = args.n_structures
    if args.n_threads:
        config.n_threads = args.n_threads
    
    # Run folding
    try:
        if args.verbose:
            print(f"DNAfold2 - DNA Structure Prediction")
            print(f"{'='*40}")
            print(f"Sequence: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
            print(f"Length: {len(sequence)} nt")
            print(f"Method: {config.sampling_method.upper()}")
            print(f"Steps: {config.folding_steps:,}")
            print(f"Na+: {config.na_concentration} mM, Mg2+: {config.mg_concentration} mM")
            print(f"Output: {args.output}")
            print()
        
        result = fold(
            sequence=sequence,
            config=config,
            output_dir=args.output,
            verbose=args.verbose
        )
        
        if args.verbose:
            print(f"\nResults saved to: {result.output_dir}")
            print(f"  CG structures: {len(result.cg_structures)}")
            print(f"  All-atom structures: {len(result.all_atom_structures)}")
            print(f"  Elapsed time: {result.elapsed_time:.1f}s")
        
        return 0
    
    except Exception as e:
        print(f"Error during folding: {e}", file=sys.stderr)
        return 1


def cmd_config(args: argparse.Namespace) -> int:
    """Execute the config command."""
    config = FoldingConfig(
        sampling_method=args.method,
        folding_steps=args.steps,
        optimizing_steps=args.optimize_steps
    )
    
    config.to_file(args.output)
    print(f"Configuration saved to: {args.output}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute the validate command."""
    try:
        sequence = validate_sequence(args.sequence)
        print(f"✓ Valid DNA sequence")
        print(f"  Length: {len(sequence)} nucleotides")
        print(f"  Composition: ", end="")
        for base in "ATCG":
            count = sequence.count(base)
            pct = 100 * count / len(sequence)
            print(f"{base}={count}({pct:.1f}%) ", end="")
        print()
        return 0
    except ValueError as e:
        print(f"✗ Invalid sequence: {e}", file=sys.stderr)
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Execute the info command."""
    from . import __version__
    from .utils import get_bin_dir, get_src_dir, check_gcc_available
    
    print(f"DNAfold2 v{__version__}")
    print(f"{'='*40}")
    print(f"Binary directory: {get_bin_dir()}")
    print(f"Source directory: {get_src_dir()}")
    print(f"GCC available: {'Yes' if check_gcc_available() else 'No'}")
    
    # Check binaries
    bin_dir = get_bin_dir()
    binaries = ["TiRNA_remc", "TiRNA_sa", "TiRNA_optimize", "op"]
    print(f"\nBinary status:")
    for b in binaries:
        path = bin_dir / b
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {b}")
    
    return 0


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.version:
        from . import __version__
        print(f"dnafold2 {__version__}")
        return 0
    
    if args.command == "fold":
        return cmd_fold(args)
    elif args.command == "config":
        return cmd_config(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "info":
        return cmd_info(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
