SF-Mamba: Spatial-Frequency Adaptive Modeling for Image Dehazing
This repository provides the official implementation of SF-Mamba, a
Spatial-Frequency adaptive modeling framework for image dehazing that
integrates structured spatial modeling with wavelet-guided expert learning.

📖 Overview
SF-Mamba addresses the limitations of existing Mamba-based dehazing methods,
which often struggle to preserve fine-grained structures and to fully leverage
the complementary structural and textural information encoded across different
frequency bands. It combines a Grouped Patch-wise Selective Scanning (GPS²)
mechanism with a standard two-dimensional selective state space model (2D-SSM)
to jointly capture complementary local and global spatial dependencies, and
introduces Wavelet-Guided Expert Learning (WGEL) to guide dynamic routing toward
specialized experts for adaptive refinement of global structures and
fine-grained textures.

The framework contains three main stages:

Structured spatial dependency modeling with Grouped Patch-wise Selective
Scanning (GPS²) and a standard 2D-SSM.

Frequency-guided expert learning with Wavelet-Guided Expert Learning
(WGEL) over low-frequency and direction-sensitive high-frequency components.

High-fidelity dehazing via adaptive refinement of contextual structures
and fine-grained details.

🧩 Components
GPS² (Grouped Patch-wise Selective Scanning): preserves local structural
continuity and improves cross-region feature communication through
hierarchical intra-patch and inter-patch interactions.

2D-SSM (Two-Dimensional Selective State Space Model): captures global
spatial dependencies complementary to GPS².

WGEL (Wavelet-Guided Expert Learning): decomposes features into one
low-frequency approximation component and three direction-sensitive
high-frequency detail components, and routes them to specialized experts.

📊 Results
Extensive experiments on both synthetic and real-world benchmarks demonstrate
that SF-Mamba consistently achieves superior dehazing performance while
maintaining competitive computational efficiency.
