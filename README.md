# OCTOPUS: OCT Optimized Processing for Ultra-precise Segmentation

## Introduction

OCTOPUS is a novel model for the segmentation of the External Limiting Membrane (ELM) in Optical Coherence Tomography (OCT) images. This project introduces a novel hybrid approach that excels in handling complex cases, particularly those involving macular holes. By combining advanced deep learning techniques with probabilistic modeling, OCTOPUS achieves unprecedented accuracy and interpretability in ELM segmentation.

## Background

Optical Coherence Tomography has revolutionized ophthalmology by providing high-resolution, cross-sectional imaging of retinal structures. The External Limiting Membrane, visible in OCT scans, is a critical biomarker for photoreceptor health. Accurate ELM segmentation, especially in the presence of macular holes, poses significant challenges due to its thin structure and potential disruptions in pathological states. OCTOPUS addresses these challenges head-on, offering a robust solution for clinicians and researchers alike.

## Key Features

- Hybrid architecture leveraging deep learning and Gaussian processes
- Specialized in ELM segmentation, with a focus on macular hole cases
- Advanced kernel analysis for optimized Gaussian Process Edge Tracing (GPET)
- State-of-the-art performance benchmarked against existing techniques
- Uncertainty quantification for enhanced clinical decision-making

## Methodology

OCTOPUS employs a two-step approach:

1. **Deep Learning Segmentation**: Utilizes a modified SegNet architecture for initial ELM delineation.
2. **Gaussian Process Refinement**: Applies GPET to refine segmentation boundaries with probabilistic edge detection.
s
## Requirements

Can be found in requirements.txt

## Usage

Refer to the demo.ipynb file for usage instructions.

## Results

OCTOPUS demonstrates superior precision and boundary accuracy, particularly in none macular hole cases, surpassing conventional segmentation methods.

## License

MIT

## Contact

uzaykaradag@outlook.com

## References

- Badrinarayanan, V. et al. SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence.
- Burke, M., King, S. (2021). Gaussian Process Edge Tracing: A Novel Approach to Boundary Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence.