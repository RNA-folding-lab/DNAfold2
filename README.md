# DNAfold2
An improved coarse-grained (CG) model for ab initio prediction of complex DNA structure folding
## Introduction
This package was developed to predict 3D structure and stability for complex DNAs in ion solutions based on the coarse-grained (CG) model developed by Xunxun Wang (Guizhou Medical University), and Ya-Zhou Shi (Wuhan Textile University), etc. This CG model was extended from the RNA CG model introduced by Ya-Zhou Shi, Lei Jin, Chen-Jie Feng, Xunxun Wang, Ya-Lan Tan and Zhi-Jie Tan at Tan's group (Wuhan University), to predict 3D structure, stability and salt effect for RNAs from their sequences.
### Model Features:
   * Three-bead coarse-grained model including sequence-dependent base-pairing/stacking potentials and structure-based electrostatic potential;
   * The model can predict the 3D structure of DNAs with multi-way junctions from their sequences;
   * The model also reproduces the thermal stability of DNA junctions across diverse sequences and lengths.
## Requirements
   * Operating System: Linux or MacOS
   * CPU: ≥ 8 cores with 2 threads
   * C++ Compiler: GCC ≥ 7.5 (with C++11 support)
   * Python: ≥ 3.11.5 and Biopython ≥ 1.81
## Usage
1. Prepare Input Files
   * Put the target sequence in seq.dat file (e.g., ATCCTAGTTATAGGAT)
   * Change the Configuration File (config.dat) including: (1) Number of steps for REMC simulations during structure folding; (2) Number of steps for energy optimization after sampling; (3) Ion concentrations of mono-/divalent ions (in mM).
2. Run Simulation
   * bash run.sh
3. Output Files
