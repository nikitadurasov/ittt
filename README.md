# IT³: Idempotent Test-Time Training


<p align="center">
  <img src="./src/method.jpg" alt="Project or Page Cover" width="99%" style="border-radius: 50px;"/>
</p>

**Idempotent Test-Time Training (IT3) Overview**  

Test-time training (TTT) refers to adapting a model during inference using only the current test input, without access to labels or additional data. Traditional TTT methods rely on domain-specific self-supervised tasks (e.g., rotation prediction), which limits their applicability. IT3 replaces these with a generic principle: **idempotence** — encouraging the model to produce stable outputs when recursively applied.

In this figure, the **training phase** (left) teaches the model `f_θ` to predict the label `y` both with and without access to it, making the function `f_θ(x, ·)` approximately idempotent. During **inference** (right), given a potentially corrupted input `x`, the model computes a first prediction `y₀ = f_θ(x, 0)`, and then a frozen copy of the model produces `y₁ = f(x, y₀)`. The model is briefly updated to minimize the difference between `y₀` and `y₁`, pulling the prediction closer to the training distribution. This simple, label-free procedure improves robustness under distribution shifts, and applies broadly across data types and model architectures.


### [Project Page](https://www.norange.io/projects/ittt/) | [arXiv Paper](https://arxiv.org/abs/2410.04201) | [ICML Paper](https://openreview.net/pdf?id=MHiTGWDbIb) | [IT3Engine [torch-ttt]](https://torch-ttt.github.io/_autosummary/torch_ttt.engine.it3_engine.IT3Engine.html)

> **For a quick tryout of the method, check the Colab notebooks below!**  
> 
> **Please also refer to the [Zigzag](https://github.com/cvlab-epfl/zigzag) and [Iterative Uncertainty](https://github.com/cvlab-epfl/iter_unc) paper, which served as the foundation for our work.**

For easy integration of `IT^3`, use the [`torch-ttt`](https://github.com/nikitadurasov/torch-ttt) library. The [`IT3Engine`](https://torch-ttt.github.io/_autosummary/torch_ttt.engine.it3_engine.IT3Engine.html#torch_ttt.engine.it3_engine.IT3Engine) class offers a seamless API for test-time training with virtually any architecture and task. You can try `IT^3` directly in this [Colab notebook](https://colab.research.google.com/github/nikitadurasov/ittt/blob/main/exps/mnist/it3_torch_ttt.ipynb).

```python
from torch_ttt.engine import IT3Engine

engine = IT3Engine(
    model=network,                # Original model
    features_layer_name="..."     # The layer to extract intermediate features from
)
```



## Video Explainer

[![Watch the video](https://img.youtube.com/vi/eKGKpN8fFRM/maxresdefault.jpg)](https://youtu.be/eKGKpN8fFRM)

### [Watch this video on YouTube](https://youtu.be/eKGKpN8fFRM)

## Experiments

### MNIST Classification

We evaluate `IT^3` on the MNIST digit classification task under Gaussian noise corruptions. During training, the model is trained on clean handwritten digits. At test time, Gaussian noise with σ=0.5 is added to each input image. While the vanilla model suffers a clear accuracy drop due to the distribution shift, `IT^3` adapts in-place at inference, leveraging its idempotence loss to stabilize predictions and significantly reduce the error rate. This shows that `IT^3` can effectively handle noise even in simple, low-dimensional domains.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikitadurasov/ittt/blob/main/exps/mnist/mnist_ittt.ipynb)

### CIFAR Classification

We repeat the same setup on CIFAR-10, a more complex and colorful image classification task. The model is trained on clean images and evaluated under test-time Gaussian noise (σ=0.2). As expected, the vanilla classifier suffers under noise, showing reduced accuracy. In contrast, `IT^3` improves prediction confidence and performance by iteratively refining outputs to match their previous predictions. These experiments highlight that our method generalizes to more complex natural image datasets and remains effective without retraining or augmenting the training data.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikitadurasov/ittt/blob/main/exps/cifar/cifar_ittt.ipynb)

### Tabular Data

We evaluate `IT^3` on tabular classification benchmarks from the UCI repository, including datasets like Adult and Covertype. After training a fully connected network on clean data, we corrupt the test features with Gaussian noise. While traditional models are brittle to such feature corruption, `IT^3` adapts at test time and improves classification accuracy. These results demonstrate that our method is not limited to vision tasks and can effectively operate in structured data domains, even with low-dimensional inputs and discrete outputs.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikitadurasov/ittt/blob/main/exps/tabular/vanilla_model.ipynb)

### Age Prediction (UTKFace)

We apply `IT^3` to a regression task on the UTKFace dataset, where the goal is to predict a person’s age from their face image. The test set is corrupted with moderate Gaussian blur and brightness shifts to simulate real-world conditions such as defocus and lighting variation. Unlike classification settings, this task requires stabilizing continuous outputs. `IT^3` is able to reduce mean absolute error (MAE) by encouraging consistent predictions across iterations, confirming that our method applies to both classification and regression tasks.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikitadurasov/ittt/blob/main/exps/age/eval_ittt.ipynb)

### Airfoils Physics

We test `IT^3` on the UCI Airfoil Self-Noise dataset, a low-dimensional physics regression problem where the task is to predict noise levels from physical features like frequency and flow velocity. Gaussian noise is added to the input features at test time. While a vanilla regression model performs poorly on the corrupted data, `IT^3` mitigates the degradation and recovers performance. This validates the method’s ability to generalize to scientific and physics-based regression domains, where interpretability and stability are crucial.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikitadurasov/ittt/blob/main/exps/airfoils/train_evaluate.ipynb)

### Road Segmentation

In this semantic segmentation task, we apply `IT^3` to aerial road images where the objective is to segment road regions from overhead imagery. The test data is synthetically corrupted with contrast changes and blur, mimicking satellite quality degradation. Standard models experience a sharp drop in IoU (Intersection-over-Union), especially on small or occluded roads. By enforcing consistency across iterative predictions, `IT^3` improves segmentation masks and restores degraded details. This showcases its utility for dense pixel-wise prediction tasks under distribution shift.

Coming soon...

### ImageNet-C Classification

We benchmark `IT^3` on ImageNet-C, a standard testbed for corruption robustness in large-scale models. We evaluate a pretrained ResNet-50 on the full set of corruptions at severity level 3 (Gaussian noise, blur, compression, weather, etc.). Despite no access to corrupted data during training, `IT^3` improves top-1 accuracy across the board by adapting to each input independently at inference. This experiment shows that `IT^3` scales well to large and deep models, and complements strong pretrained baselines with minimal overhead.

Coming soon...


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