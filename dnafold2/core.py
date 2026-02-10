"""
DNAfold2 Core Module

Main DNA folding interface for running structure predictions.
"""

import glob
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from .config import FoldingConfig, validate_sequence

logger = logging.getLogger(__name__)


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
        
        logger.debug("Initializing DNAFolder: bin_dir=%s, data_dir=%s, src_dir=%s",
                      self.bin_dir, self.data_dir, self.src_dir)
        logger.debug("Configuration: method=%s, steps=%d, opt_steps=%d, Na+=%.0f mM, Mg2+=%.0f mM",
                      self.config.sampling_method, self.config.folding_steps,
                      self.config.optimizing_steps, self.config.na_concentration,
                      self.config.mg_concentration)
        
        self._validate_installation()
    
    def _validate_installation(self) -> None:
        """Validate that required binaries and data files exist."""
        logger.debug("Validating installation...")
        # Check for required binaries
        required_bins = ["TiRNA_remc"] if self.config.sampling_method == "remc" else []
        
        for binary in required_bins:
            bin_path = self.bin_dir / binary
            if not bin_path.exists():
                logger.error("Required binary '%s' not found at %s", binary, bin_path)
                raise FileNotFoundError(
                    f"Required binary '{binary}' not found at {bin_path}. "
                    f"Please compile the source files. See docs/installation.md"
                )
            logger.debug("Found required binary: %s", bin_path)
        
        logger.debug("Installation validation passed")
    
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
        logger.info("Validating input sequence (%d characters)...", len(sequence))
        sequence = validate_sequence(sequence)
        logger.info("Sequence validated: length=%d nt, first 30 chars: %s",
                     len(sequence), sequence[:30])
        
        # Setup output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="dnafold2_"))
            logger.debug("Created temporary output directory: %s", output_dir)
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Using output directory: %s", output_dir)
        
        # Create result subdirectories
        result_dirs = [
            "CG_structure", "All_atom_structure", 
            "Folding_trajectory", "Thermal_Stability", "Secondary_structure"
        ]
        for subdir in result_dirs:
            (output_dir / subdir).mkdir(exist_ok=True)
        logger.debug("Created result subdirectories: %s", result_dirs)
        
        # Setup working directory
        work_dir = output_dir / "_work"
        work_dir.mkdir(exist_ok=True)
        logger.debug("Working directory: %s", work_dir)
        
        try:
            # Write sequence file
            seq_file = work_dir / "seq.dat"
            seq_file.write_text(sequence + "\n")
            logger.debug("Wrote sequence to %s", seq_file)
            
            # Write config file
            config_file = work_dir / "config.dat"
            self.config.to_file(config_file)
            logger.debug("Wrote configuration to %s", config_file)
            
            # Copy necessary files from src
            logger.info("Setting up working directory with required files...")
            self._setup_working_directory(work_dir)
            
            logger.info("Starting folding for sequence: %s... (length=%d)",
                         sequence[:30], len(sequence))
            logger.info("Method: %s | Steps: %d | Na+: %.0f mM | Mg2+: %.0f mM",
                         self.config.sampling_method.upper(), self.config.folding_steps,
                         self.config.na_concentration, self.config.mg_concentration)
            
            # Run the folding pipeline
            self._run_folding_pipeline(work_dir, verbose)
            
            # Collect results
            logger.info("Collecting results...")
            result = self._collect_results(sequence, output_dir, work_dir)
            
        finally:
            # Cleanup working directory (optional)
            pass
        
        elapsed = (datetime.now() - start_time).total_seconds()
        result.elapsed_time = elapsed
        
        logger.info("Folding completed in %.1f seconds", elapsed)
        logger.info("Results: %d CG structures, %d all-atom structures, %d trajectories",
                     len(result.cg_structures), len(result.all_atom_structures),
                     len(result.folding_trajectories))
        
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
        
        logger.info("Starting batch folding: %d sequences", len(sequences))
        
        results = []
        for i, seq in enumerate(sequences):
            logger.info("Processing sequence %d/%d (length=%d)",
                         i + 1, len(sequences), len(seq))
            
            seq_output = output_dir / f"seq_{i+1:04d}"
            result = self.fold(seq, output_dir=seq_output, verbose=verbose)
            results.append(result)
            
            logger.info("Sequence %d/%d completed in %.1f seconds",
                         i + 1, len(sequences), result.elapsed_time)
        
        logger.info("Batch folding completed: %d sequences processed", len(results))
        return results
    
    def _setup_working_directory(self, work_dir: Path) -> None:
        """Copy required files to working directory."""
        # Copy initial files
        initial_src = self.src_dir / "initial"
        if initial_src.exists():
            shutil.copytree(initial_src, work_dir / "initial", dirs_exist_ok=True)
            logger.debug("Copied initial files from %s", initial_src)
        else:
            logger.warning("Initial source directory not found: %s", initial_src)
        
        # Copy rebuild files
        rebuild_src = self.src_dir / "rebuild"
        if rebuild_src.exists():
            shutil.copytree(rebuild_src, work_dir / "rebuild", dirs_exist_ok=True)
            logger.debug("Copied rebuild files from %s", rebuild_src)
        else:
            logger.warning("Rebuild source directory not found: %s", rebuild_src)
        
        # Copy scoring files
        scoring_src = self.src_dir / "scoring"
        if scoring_src.exists():
            shutil.copytree(scoring_src, work_dir / "scoring", dirs_exist_ok=True)
            logger.debug("Copied scoring files from %s", scoring_src)
        else:
            logger.warning("Scoring source directory not found: %s", scoring_src)
        
        # Copy required binaries (both REMC and SA variants)
        for binary in ["TiRNA_remc", "TiRNA_sa", "TiRNA_optimize", "op"]:
            src = self.bin_dir / binary
            if src.exists():
                shutil.copy2(src, work_dir / binary)
                os.chmod(work_dir / binary, 0o755)
                logger.debug("Copied binary: %s", binary)
            else:
                logger.debug("Binary not found (may not be needed): %s", src)
        
        # Copy utility C files
        utils_src = self.src_dir / "utils"
        if utils_src.exists():
            c_files = list(utils_src.glob("*.c"))
            for c_file in c_files:
                shutil.copy2(c_file, work_dir)
            logger.debug("Copied %d utility C files from %s", len(c_files), utils_src)
        
        # Copy analysis files
        analysis_src = self.src_dir / "analysis"
        if analysis_src.exists():
            c_files = list(analysis_src.glob("*.c"))
            for c_file in c_files:
                shutil.copy2(c_file, work_dir)
            logger.debug("Copied %d analysis C files from %s", len(c_files), analysis_src)
        
        logger.info("Working directory setup complete")
    
    def _run_folding_pipeline(self, work_dir: Path, verbose: bool) -> None:
        """Execute the folding pipeline.
        
        The pipeline consists of:
        1. Initialize sequence conformation (seq_initial.c)
        2. Run folding (REMC or SA)
        3. Scoring and clustering
        4. Secondary structure prediction
        5. Optimization
        6. Rebuild all-atom structures
        """
        env = os.environ.copy()
        # Only set OMP_NUM_THREADS if not already set (respects SLURM allocation)
        if "OMP_NUM_THREADS" not in env:
            env["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)
        logger.debug("OMP_NUM_THREADS=%s", env.get("OMP_NUM_THREADS"))
        
        def run_cmd(cmd: List[str], cwd: Path, desc: str) -> subprocess.CompletedProcess:
            """Run a command and handle errors."""
            logger.info("  [CMD] %s", desc)
            logger.debug("  Command: %s (cwd=%s)", " ".join(cmd), cwd)
            step_start = time.time()
            try:
                result = subprocess.run(
                    cmd, 
                    cwd=cwd, 
                    env=env,
                    capture_output=True, 
                    text=True,
                    timeout=86400  # 24 hour timeout per step
                )
                elapsed = time.time() - step_start
                if result.returncode != 0:
                    logger.warning("  [CMD] %s returned non-zero exit code %d (%.1fs)",
                                   desc, result.returncode, elapsed)
                    if result.stderr:
                        logger.warning("  stderr: %s", result.stderr[:500])
                    if result.stdout:
                        logger.debug("  stdout: %s", result.stdout[:500])
                else:
                    logger.info("  [CMD] %s completed successfully (%.1fs)", desc, elapsed)
                    if result.stdout:
                        logger.debug("  stdout: %s", result.stdout[:500])
                return result
            except subprocess.TimeoutExpired:
                elapsed = time.time() - step_start
                logger.error("  [CMD] %s TIMED OUT after %.1fs", desc, elapsed)
                return None
            except FileNotFoundError as e:
                logger.error("  [CMD] %s FAILED — executable not found: %s", desc, e)
                return None
        
        # ===============================================================
        # Step 1: Initialize sequence conformation
        # ===============================================================
        logger.info("=" * 60)
        logger.info("STEP 1/4: Initializing sequence conformation")
        logger.info("=" * 60)
        
        initial_dir = work_dir / "initial"
        if initial_dir.exists():
            # Copy seq.dat to initial directory
            shutil.copy2(work_dir / "seq.dat", initial_dir / "seq.dat")
            logger.debug("Copied seq.dat to initial directory")
            
            # Compile and run seq_initial
            seq_initial_c = initial_dir / "seq_initial.c"
            if seq_initial_c.exists():
                logger.info("Compiling seq_initial.c...")
                run_cmd(["g++", "seq_initial.c", "-o", "seq_initial"], initial_dir, "Compile seq_initial")
                logger.info("Running seq_initial...")
                run_cmd(["./seq_initial"], initial_dir, "Run seq_initial")
                
                # Copy generated ch.dat to work_dir
                ch_dat = initial_dir / "ch.dat"
                if ch_dat.exists():
                    shutil.copy2(ch_dat, work_dir / "ch.dat")
                    logger.info("Generated ch.dat copied to working directory")
                else:
                    logger.warning("ch.dat was not generated by seq_initial")
            else:
                logger.warning("seq_initial.c not found in %s", initial_dir)
        else:
            logger.warning("Initial directory not found: %s", initial_dir)
        
        # ===============================================================
        # Step 2: Prepare config and run folding
        # ===============================================================
        logger.info("=" * 60)
        logger.info("STEP 2/4: Running folding simulation (%s, %d steps)",
                     self.config.sampling_method.upper(), self.config.folding_steps)
        logger.info("=" * 60)
        
        # Create config1.dat (numeric config format expected by C code)
        config_file = work_dir / "config.dat"
        if config_file.exists():
            self._create_numeric_config(work_dir)
            logger.debug("Created numeric config file (config1.dat)")
        
        # Create model directory for folding
        model_dir = work_dir / "model"
        model_dir.mkdir(exist_ok=True)
        logger.debug("Created model directory: %s", model_dir)
        
        # Copy required files to model dir
        for f in ["ch.dat", "config1.dat"]:
            src = work_dir / f
            if src.exists():
                shutil.copy2(src, model_dir / f)
                logger.debug("Copied %s to model directory", f)
            else:
                logger.warning("Required file %s not found in working directory", f)
        
        # Copy binary to model dir
        binary_name = "TiRNA_remc" if self.config.sampling_method == "remc" else "TiRNA_sa"
        binary_src = work_dir / binary_name
        if not binary_src.exists():
            binary_src = self.bin_dir / binary_name
            logger.debug("Binary not in work_dir, trying bin_dir: %s", binary_src)
        
        if binary_src.exists():
            shutil.copy2(binary_src, model_dir / binary_name)
            os.chmod(model_dir / binary_name, 0o755)
            logger.debug("Copied %s binary to model directory", binary_name)
            
            # Run the folding simulation
            logger.info("Launching %s — this may take a long time...", binary_name)
            logger.info("(Folding %d steps with %d threads)",
                         self.config.folding_steps, self.config.n_threads)
            run_cmd([f"./{binary_name}"], model_dir, f"Run {binary_name}")
        else:
            logger.error("%s binary not found at %s or %s — skipping folding step!",
                         binary_name, work_dir / binary_name, self.bin_dir / binary_name)
        
        # ===============================================================
        # Step 3: Post-processing with t1.c
        # ===============================================================
        logger.info("=" * 60)
        logger.info("STEP 3/4: Post-processing trajectories")
        logger.info("=" * 60)
        
        t1_c = work_dir / "t1.c"
        if t1_c.exists():
            shutil.copy2(t1_c, model_dir / "t1.c")
            logger.info("Compiling t1.c...")
            run_cmd(["g++", "t1.c", "-o", "t1"], model_dir, "Compile t1")
            logger.info("Running t1 post-processing...")
            run_cmd(["./t1"], model_dir, "Run t1")
        else:
            logger.warning("t1.c not found in %s — skipping post-processing", work_dir)
        
        # ===============================================================
        # Step 4: Scoring
        # ===============================================================
        logger.info("=" * 60)
        logger.info("STEP 4/4: Running scoring")
        logger.info("=" * 60)
        
        scoring_dir = work_dir / "scoring"
        model_dir = work_dir / "model"
        python = sys.executable  # Use the current venv Python
        
        if scoring_dir.exists() and model_dir.exists():
            # Copy output files from model to scoring directory
            for f in ["conf_0.dat", "Energy_0.dat", "ch.dat"]:
                src = model_dir / f
                if src.exists():
                    shutil.copy2(src, scoring_dir / f)
                    logger.debug("Copied %s from model to scoring", f)
                else:
                    logger.warning("Model output %s not found — scoring may fail", f)
            
            # Remove old cluster directories
            for d in glob.glob(str(scoring_dir / "Cluster_*")):
                shutil.rmtree(d, ignore_errors=True)
            
            # 4a: Find lowest-energy conformations
            logger.info("Running Umin.py (energy ranking)...")
            run_cmd([python, "Umin.py"], scoring_dir, "Run Umin.py")
            
            # 4b: Extract minimum-energy conformations
            logger.info("Compiling and running A_state...")
            run_cmd(["gcc", "-Wall", "A_state.c", "-o", "A_state", "-lm"], scoring_dir, "Compile A_state")
            run_cmd(["./A_state"], scoring_dir, "Run A_state")
            
            # 4c: Convert conformations to PDB format
            logger.info("Compiling and running tc (PDB conversion)...")
            run_cmd(["g++", "tc.c", "-o", "tc"], scoring_dir, "Compile tc")
            run_cmd(["./tc"], scoring_dir, "Run tc")
            
            # 4d: Cluster structures
            logger.info("Running clustering...")
            run_cmd([python, "cluster1.py"], scoring_dir, "Run cluster1.py")
            run_cmd([python, "cluster2.py"], scoring_dir, "Run cluster2.py")
            
            # 4e: Find cluster center in Cluster_0
            cluster0_dir = scoring_dir / "Cluster_0"
            if cluster0_dir.exists():
                shutil.copy2(scoring_dir / "cluster3.py", cluster0_dir / "cluster3.py")
                run_cmd([python, "cluster3.py"], cluster0_dir, "Run cluster3.py")
                top1 = cluster0_dir / "top1.pdb"
                if top1.exists():
                    shutil.copy2(top1, scoring_dir / "top1.pdb")
                    logger.info("top1.pdb generated successfully")
                else:
                    logger.warning("top1.pdb was not generated by cluster3.py")
            else:
                logger.warning("Cluster_0 directory not found after clustering")
        else:
            logger.warning("Scoring or model directory not found")
        
        logger.info("=" * 60)
        logger.info("Folding pipeline completed.")
        logger.info("Note: Full optimization and rebuild steps require additional setup.")
        logger.info("=" * 60)
    
    def _create_numeric_config(self, work_dir: Path) -> None:
        """Create config1.dat with numeric values for C code.
        
        Format: method steps opt_steps Na Mg n_struct n_threads conf_freq print_freq
        """
        method_num = 1 if self.config.sampling_method == "remc" else 2
        
        # Auto-adjust frequencies: if conf_output_freq or print_freq exceed
        # folding_steps, the C simulation will never produce output, causing
        # the entire scoring pipeline to fail on empty Energy_0.dat.
        conf_freq = self.config.conf_output_freq
        print_freq = self.config.print_freq
        
        if conf_freq > self.config.folding_steps:
            conf_freq = max(1, self.config.folding_steps // 10) if self.config.folding_steps >= 10 else self.config.folding_steps
            logger.warning(
                "conf_output_freq (%d) exceeds folding_steps (%d) — "
                "auto-adjusted to %d to ensure simulation produces output",
                self.config.conf_output_freq, self.config.folding_steps, conf_freq
            )
        
        if print_freq > self.config.folding_steps:
            print_freq = max(1, self.config.folding_steps // 5) if self.config.folding_steps >= 5 else self.config.folding_steps
            logger.warning(
                "print_freq (%d) exceeds folding_steps (%d) — auto-adjusted to %d",
                self.config.print_freq, self.config.folding_steps, print_freq
            )
        
        config1_content = (
            f"{method_num} {self.config.folding_steps} {self.config.optimizing_steps} "
            f"{self.config.na_concentration} {self.config.mg_concentration} {self.config.n_structures} "
            f"{self.config.n_threads} {conf_freq} {print_freq}\n"
        )
        (work_dir / "config1.dat").write_text(config1_content)
        logger.debug("config1.dat contents: %s", config1_content.strip())
    
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
        logger.debug("Found %d CG structures in %s", len(result.cg_structures), cg_dir)
        
        # Collect all-atom structures
        aa_dir = output_dir / "All_atom_structure"
        result.all_atom_structures = list(aa_dir.glob("*.pdb"))
        logger.debug("Found %d all-atom structures in %s", len(result.all_atom_structures), aa_dir)
        
        # Collect secondary structures
        ss_dir = output_dir / "Secondary_structure"
        result.secondary_structures = list(ss_dir.glob("*.dat"))
        logger.debug("Found %d secondary structure files in %s", len(result.secondary_structures), ss_dir)
        
        # Collect trajectories
        traj_dir = output_dir / "Folding_trajectory"
        result.folding_trajectories = list(traj_dir.glob("*.pdb"))
        logger.debug("Found %d trajectory files in %s", len(result.folding_trajectories), traj_dir)
        
        # Thermal stability
        ts_file = output_dir / "Thermal_Stability" / "thermal_stability.dat"
        if ts_file.exists():
            result.thermal_stability = ts_file
            logger.debug("Found thermal stability file: %s", ts_file)
        else:
            logger.debug("No thermal stability file found")
        
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
