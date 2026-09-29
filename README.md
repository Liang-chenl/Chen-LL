SF-Mamba: Spatial-Frequency Adaptive Modeling for Image Dehazing
Authors: 未提供

https://img.shields.io/badge/Paper-%3CCOLOR%3E.svg

Notice: This repository provides the official implementation of SF-Mamba. The complete code will be made publicly available after publication.

<hr />
Abstract: Existing Mamba-based dehazing methods often struggle to preserve fine-grained structures and to fully leverage the complementary structural and textural information encoded across different frequency bands. To address these limitations, we propose SF-Mamba, a Spatial-Frequency adaptive modeling framework for image dehazing that integrates structured spatial modeling with wavelet-guided expert learning. Specifically, SF-Mamba combines a Grouped Patch-wise Selective Scanning (GPS²) mechanism with a standard two-dimensional selective state space model (2D-SSM) to jointly capture complementary local and global spatial dependencies. In addition, it introduces Wavelet-Guided Expert Learning (WGEL) to guide dynamic routing toward specialized experts for adaptive refinement of global structures and fine-grained textures. Extensive experiments on both synthetic and real-world benchmarks demonstrate that SF-Mamba consistently achieves superior dehazing performance while maintaining competitive computational efficiency.

<hr />
📖 Overview
SF-Mamba is a Spatial-Frequency adaptive modeling framework for image dehazing that integrates structured spatial modeling with wavelet-guided expert learning.

It addresses the limitations of existing Mamba-based dehazing methods, which often struggle to preserve fine-grained structures and to fully leverage the complementary structural and textural information encoded across different frequency bands. The framework combines a Grouped Patch-wise Selective Scanning (GPS²) mechanism with a standard two-dimensional selective state space model (2D-SSM) to jointly capture complementary local and global spatial dependencies, and introduces Wavelet-Guided Expert Learning (WGEL) to guide dynamic routing toward specialized experts for adaptive refinement of global structures and fine-grained textures.

The framework contains three main stages:

Structured spatial dependency modeling with Grouped Patch-wise Selective Scanning (GPS²) and a standard 2D-SSM.

Frequency-guided expert learning with Wavelet-Guided Expert Learning (WGEL) over low-frequency and direction-sensitive high-frequency components.

High-fidelity dehazing via adaptive refinement of contextual structures and fine-grained details.

<hr />
🧩 Components
GPS² (Grouped Patch-wise Selective Scanning): Preserves local structural continuity and improves cross-region feature communication through hierarchical intra-patch and inter-patch interactions.

2D-SSM (Two-Dimensional Selective State Space Model): Captures global spatial dependencies complementary to GPS².

WGEL (Wavelet-Guided Expert Learning): Decomposes features into one low-frequency approximation component and three direction-sensitive high-frequency detail components, and routes them to specialized experts.

<hr />
📊 Results
Extensive experiments on both synthetic and real-world benchmarks demonstrate that SF-Mamba consistently achieves superior dehazing performance while maintaining competitive computational efficiency.

<hr />
🚧 Code Availability
This repository provides the official implementation of SF-Mamba.

Notice: The complete code will be made publicly available after publication.

<hr />
