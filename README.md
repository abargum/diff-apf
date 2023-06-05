# Diff-APF
**Differentiable Allpass Filters for Phase Response Estimation and Automatic Signal Alignment**

This is the official PyTorch exemplification repository of ***Bargum et al.* (Aalborg University), [Diff APF](https://arxiv.org/abs/2306.00860)**.

Audio samples can be heard through the link below!

[![arXiv](https://img.shields.io/badge/arXiv-2106.07889-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2306.00860) [![githubio](https://img.shields.io/static/v1?message=Audio%20Samples&logo=Github&labelColor=grey&color=blue&logoColor=white&label=%20&style=flat-square)](https://abargum.github.io/)

## Abstract

Virtual analog (VA) audio effects are increasingly based on neural networks and deep learning frameworks. Due to the underlying black-box methodology, a successful model will learn to approximate the data it is presented, including potential errors such as latency and audio dropouts as well as non-linear characteristics and frequency-dependent phase shifts produced by the hardware. The latter is of particular interest as the learned phase-response might cause unwanted audible artifacts when the effect is used for creative processing techniques such as dry-wet mixing or parallel compression. To overcome these artifacts we propose differentiable signal processing tools and deep optimization structures for automatically tuning all-pass filters to predict the phase response of different VA simulations, and align processed signals that are out of phase. The approaches are assessed using objective metrics while listening tests evaluate their ability to enhance the quality of parallel path processing techniques. Ultimately, an over-parameterized, BiasNet-based, all-pass model is proposed for the optimization problem under consideration, resulting in models that can estimate all-pass filter coefficients to align a dry signal with its affected, wet, equivalent.

## Notes

- Clone the repository and run the notebook to either train your own models or use the pretrained models presented in the paper.
- Use section 1. for personal experimentation and section 2. for inference with the pretrained models.
- Create and train cascades of traditional, warped and warped + strethed allpass filters to obtain the desired filter order.
- Experiment with different loss functions, intialized in the losses.py script

## References and Building Blocks

- *J. Steinmetz and J. D. Reiss, “auraloss: Audio focused loss functions in PyTorch,” in Digital Music Research
Network One-day Workshop (DMRN+15), 2020.
- *G. Pepe, L. Gabrielli, S. Squartini, C. Tripodi, and N. Strozzi, “Deep optimization of parametric IIR filters
for audio equalization,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp.
1136–1149, 2022.
- *Kuznetsov, J. D. Parker, and F. Esqueda, “Differentiable IIR filters for machine learning applications,” in
Proc. Intl. Conf. Digital Audio Effects, 2020, pp. 165–172.






