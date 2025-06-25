# IT³: Idempotent Test-Time Training



<!-- <p align="center">
  <img src="./src/method.jpg" alt="Project or Page Cover" width="90%" style="border-radius: 50px;"/>
</p>

**Idempotent Test-Time Training (IT3) approach:** During training (left), the model `f_θ` is trained to predict the label `y` with or without `y` given to it as input. At test time (right), when given a corrupted input, the model is applied sequentially. It then briefly trains with the objective of making `f_θ(x, ·)` idempotent, using only the current test input. -->



<!-- [![arXiv](https://img.shields.io/badge/cs.CV-arXiv%3A2410.04201-blue?logo=arxiv&color=red)](https://arxiv.org/abs/2410.04201)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&color=blue)](https://www.python.org/downloads/release/python-31014/)
[![Pytorch](https://img.shields.io/badge/Pytorch-2.2.1-blue?logo=pytorch&color=blue)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/nikitadurasov/ittt/blob/main/LICENSE) -->

### [Project Page](https://www.norange.io/projects/ittt/) | [arXiv Paper](https://arxiv.org/abs/2410.04201)

> **For a quick tryout of the method, check the Colab notebooks below!**

> **Please also refer to the [Zigzag](https://github.com/cvlab-epfl/zigzag) and [Iterative Uncertainy](https://github.com/cvlab-epfl/iter_unc) paper, which served as the foundation for our work.**

## Video Explainer

[![Watch the video](https://img.youtube.com/vi/eKGKpN8fFRM/maxresdefault.jpg)](https://youtu.be/eKGKpN8fFRM)

### [Watch this video on YouTube](https://youtu.be/eKGKpN8fFRM)

## Experiments

### MNIST Classification

We train the networks on the MNIST training set and evaluate both the vanilla model and our Idempotent Test-Time Training (`IT^3`) approach on the test set with added Gaussian noise. As expected, the vanilla model shows a significant drop in performance compared to its results on the clean test data. In contrast, our `IT^3` approach demonstrates better results with smaller accuracy degradation, showcasing its effectiveness in handling noisy inputs.


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikitadurasov/ittt/blob/main/exps/mnist/mnist_ittt.ipynb)

### CIFAR Classification

We train the networks on the CIFAR training set and evaluate both the vanilla model and our Idempotent Test-Time Training (`IT^3`) approach on a Gaussian-noised test set, similar to our experiments with MNIST. As expected, the vanilla model experiences a substantial drop in performance compared to its results on the clean test data. In contrast, our `IT^3` approach achieves better results with less accuracy degradation, highlighting its effectiveness in handling noisy inputs.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikitadurasov/ittt/blob/main/exps/cifar/cifar_ittt.ipynb)

### Tabular Data

Experiment description

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikitadurasov/ittt/blob/main/exps/tabular/vanilla_model.ipynb)

### Age Prediction

Experiment description

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikitadurasov/ittt/blob/main/exps/age/eval_ittt.ipynb)

### Airfoils Physics

Experiment description

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikitadurasov/ittt/blob/main/exps/airfoils/train_evaluate.ipynb)

More coming soon...

## Citation

If you find this code useful, please consider citing our paper:

> Durasov, Nikita, et al. "IT³: Idempotent Test-Time Training." ICML 2025.

```bibtex
@article{durasov2025it,
  title={{IT}\${\textasciicircum}3\$: Idempotent Test-Time Training},
  author={Nikita Durasov and Assaf Shocher and Doruk Oner and Gal Chechik and Alexei A Efros and Pascal Fua}, 
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=MHiTGWDbIb}
}
``` 
