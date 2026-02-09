"""
DNAfold2 Core Module

Main DNA folding interface for running structure predictions.
"""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from .config import FoldingConfig, validate_sequence


@dataclass
class FoldingResult:
    """Results from a DNA folding simulation.
    
    Attributes:
        sequence: The input DNA sequence
        cg_structures: Paths to coarse-grained structure PDB files
        all_atom_structures: Paths to all-atom structure PDB files
        secondary_structures: Paths to secondary structure files (dot-bracket notation)
        folding_trajectories: Paths to folding trajectory files
        thermal_stability: Path to thermal stability data file
        output_dir: Directory containing all output files
        elapsed_time: Total runtime in seconds
    """
    sequence: str
    cg_structures: List[Path] = field(default_factory=list)
    all_atom_structures: List[Path] = field(default_factory=list)
    secondary_structures: List[Path] = field(default_factory=list)
    folding_trajectories: List[Path] = field(default_factory=list)
    thermal_stability: Optional[Path] = None
    output_dir: Optional[Path] = None
    elapsed_time: float = 0.0
    
    def __repr__(self) -> str:
        return (f"FoldingResult(sequence='{self.sequence[:20]}...', "
                f"n_structures={len(self.cg_structures)}, "
                f"output_dir='{self.output_dir}')")


class DNAFolder:
    """Main class for DNA structure folding using the CG model.
    
    This class provides a Python interface to the DNAfold2 C-based folding
    algorithms, supporting both Replica Exchange Monte Carlo (REMC) and
    Simulated Annealing (SA) sampling methods.
    
    Example:
        >>> from dnafold2 import DNAFolder, FoldingConfig
        >>> 
        >>> config = FoldingConfig(
        ...     sampling_method="remc",
        ...     folding_steps=500000,
        ...     na_concentration=1000
        ... )
        >>> folder = DNAFolder(config)
        >>> result = folder.fold("ATCCTAGTTATAGGAT")
        >>> print(result.cg_structures)
    """
    
    def __init__(
        self, 
        config: Optional[FoldingConfig] = None,
        bin_dir: Optional[str | Path] = None,
        data_dir: Optional[str | Path] = None
    ):
        """Initialize the DNA folder.
        
        Args:
            config: Folding configuration. Uses defaults if not provided.
            bin_dir: Directory containing compiled binaries. Auto-detected if not specified.
            data_dir: Directory containing fragment libraries and data. Auto-detected if not specified.
        """
        self.config = config or FoldingConfig()
        
        # Locate package directories
        package_root = Path(__file__).parent.parent
        self.bin_dir = Path(bin_dir) if bin_dir else package_root / "bin"
        self.data_dir = Path(data_dir) if data_dir else package_root / "data"
        self.src_dir = package_root / "src"
        
        self._validate_installation()
    
    def _validate_installation(self) -> None:
        """Validate that required binaries and data files exist."""
        # Check for required binaries
        required_bins = ["TiRNA_remc"] if self.config.sampling_method == "remc" else []
        
        for binary in required_bins:
            bin_path = self.bin_dir / binary
            if not bin_path.exists():
                raise FileNotFoundError(
                    f"Required binary '{binary}' not found at {bin_path}. "
                    f"Please compile the source files. See docs/installation.md"
                )
    
    def fold(
        self, 
        sequence: str, 
        output_dir: Optional[str | Path] = None,
        verbose: bool = False
    ) -> FoldingResult:
        """Fold a DNA sequence to predict its 3D structure.
        
        Args:
            sequence: DNA sequence string (only A, T, C, G allowed)
            output_dir: Directory to save results. Uses temp dir if not specified.
            verbose: If True, print progress information.
            
        Returns:
            FoldingResult containing paths to output files
            
        Raises:
            ValueError: If sequence is invalid
            RuntimeError: If folding fails
        """
        start_time = datetime.now()
        
        # Validate sequence
        sequence = validate_sequence(sequence)
        
        # Setup output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="dnafold2_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create result subdirectories
        result_dirs = [
            "CG_structure", "All_atom_structure", 
            "Folding_trajectory", "Thermal_Stability", "Secondary_structure"
        ]
        for subdir in result_dirs:
            (output_dir / subdir).mkdir(exist_ok=True)
        
        # Setup working directory
        work_dir = output_dir / "_work"
        work_dir.mkdir(exist_ok=True)
        
        try:
            # Write sequence file
            seq_file = work_dir / "seq.dat"
            seq_file.write_text(sequence + "\n")
            
            # Write config file
            config_file = work_dir / "config.dat"
            self.config.to_file(config_file)
            
            # Copy necessary files from src
            self._setup_working_directory(work_dir)
            
            if verbose:
                print(f"Starting folding for sequence: {sequence[:30]}...")
                print(f"Method: {self.config.sampling_method.upper()}")
                print(f"Folding steps: {self.config.folding_steps}")
            
            # Run the folding pipeline
            self._run_folding_pipeline(work_dir, verbose)
            
            # Collect results
            result = self._collect_results(sequence, output_dir, work_dir)
            
        finally:
            # Cleanup working directory (optional)
            pass
        
        elapsed = (datetime.now() - start_time).total_seconds()
        result.elapsed_time = elapsed
        
        if verbose:
            print(f"Folding completed in {elapsed:.1f} seconds")
        
        return result
    
    def fold_batch(
        self, 
        sequences: List[str],
        output_dir: str | Path,
        verbose: bool = False
    ) -> List[FoldingResult]:
        """Fold multiple DNA sequences.
        
        Args:
            sequences: List of DNA sequence strings
            output_dir: Base directory for all results
            verbose: If True, print progress information
            
        Returns:
            List of FoldingResult objects
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        for i, seq in enumerate(sequences):
            if verbose:
                print(f"\nProcessing sequence {i+1}/{len(sequences)}")
            
            seq_output = output_dir / f"seq_{i+1:04d}"
            result = self.fold(seq, output_dir=seq_output, verbose=verbose)
            results.append(result)
        
        return results
    
    def _setup_working_directory(self, work_dir: Path) -> None:
        """Copy required files to working directory."""
        # Copy initial files
        initial_src = self.src_dir / "initial"
        if initial_src.exists():
            shutil.copytree(initial_src, work_dir / "initial", dirs_exist_ok=True)
        
        # Copy rebuild files
        rebuild_src = self.src_dir / "rebuild"
        if rebuild_src.exists():
            shutil.copytree(rebuild_src, work_dir / "rebuild", dirs_exist_ok=True)
        
        # Copy scoring files
        scoring_src = self.src_dir / "scoring"
        if scoring_src.exists():
            shutil.copytree(scoring_src, work_dir / "scoring", dirs_exist_ok=True)
        
        # Copy required binaries
        for binary in ["TiRNA_remc", "op"]:
            src = self.bin_dir / binary
            if src.exists():
                shutil.copy2(src, work_dir / binary)
                os.chmod(work_dir / binary, 0o755)
        
        # Copy utility C files
        utils_src = self.src_dir / "utils"
        if utils_src.exists():
            for c_file in utils_src.glob("*.c"):
                shutil.copy2(c_file, work_dir)
        
        # Copy analysis files
        analysis_src = self.src_dir / "analysis"
        if analysis_src.exists():
            for c_file in analysis_src.glob("*.c"):
                shutil.copy2(c_file, work_dir)
    
    def _run_folding_pipeline(self, work_dir: Path, verbose: bool) -> None:
        """Execute the folding pipeline."""
        # This would run the actual C executables
        # For now, we'll use subprocess to call the binaries
        
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)
        
        # The actual implementation would:
        # 1. Run initial.sh equivalent (sequence initialization)
        # 2. Run TiRNA_remc or TiRNA_sa
        # 3. Run scoring
        # 4. Run secondary structure prediction
        # 5. Run optimization
        # 6. Run rebuild
        
        if verbose:
            print("  Note: Full pipeline execution requires compiled binaries")
    
    def _collect_results(
        self, 
        sequence: str, 
        output_dir: Path, 
        work_dir: Path
    ) -> FoldingResult:
        """Collect and organize output files."""
        result = FoldingResult(sequence=sequence, output_dir=output_dir)
        
        # Collect CG structures
        cg_dir = output_dir / "CG_structure"
        result.cg_structures = list(cg_dir.glob("*.pdb"))
        
        # Collect all-atom structures
        aa_dir = output_dir / "All_atom_structure"
        result.all_atom_structures = list(aa_dir.glob("*.pdb"))
        
        # Collect secondary structures
        ss_dir = output_dir / "Secondary_structure"
        result.secondary_structures = list(ss_dir.glob("*.dat"))
        
        # Collect trajectories
        traj_dir = output_dir / "Folding_trajectory"
        result.folding_trajectories = list(traj_dir.glob("*.pdb"))
        
        # Thermal stability
        ts_file = output_dir / "Thermal_Stability" / "thermal_stability.dat"
        if ts_file.exists():
            result.thermal_stability = ts_file
        
        return result


def fold(
    sequence: str,
    config: Optional[FoldingConfig] = None,
    output_dir: Optional[str | Path] = None,
    verbose: bool = False
) -> FoldingResult:
    """Convenience function to fold a DNA sequence.
    
    Args:
        sequence: DNA sequence string
        config: Folding configuration (uses defaults if not provided)
        output_dir: Directory to save results
        verbose: If True, print progress
        
    Returns:
        FoldingResult containing output file paths
    """
    folder = DNAFolder(config=config)
    return folder.fold(sequence, output_dir=output_dir, verbose=verbose)
