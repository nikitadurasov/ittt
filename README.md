# IT³: Idempotent Test-Time Training


<p align="center">
  <img src="./src/method.jpg" alt="Project or Page Cover" width="99%" style="border-radius: 50px;"/>
</p>

Test-time training (TTT) refers to adapting a model during inference using only the current test input, without access to labels or additional data. Traditional TTT methods rely on domain-specific self-supervised tasks (e.g., rotation prediction), which limits their applicability. IT3 replaces these with a generic principle: **idempotence** — encouraging the model to produce stable outputs when recursively applied.

In IT³, the **training phase** (left) trains the model `f_θ` to predict the label `y` both with and without access to it, making the function `f_θ(x, ·)` approximately idempotent. During **inference** (right), given a potentially corrupted or out-of-distribution input `x`, the model computes a first prediction `y₀ = f_θ(x, 0)`, and then the model also produces `y₁ = f(x, y₀)`. The model is briefly optimized to minimize the difference between `y₀` and `y₁`, pulling the prediction closer to each other and to training distribution. This simple, label-free procedure improves robustness under distribution shifts, and applies broadly across data types and model architectures.


### [Project Page](https://www.norange.io/projects/ittt/) | [arXiv Paper](https://arxiv.org/abs/2410.04201) | [ICML Paper](https://openreview.net/pdf?id=MHiTGWDbIb) | [IT3Engine [torch-ttt]](https://torch-ttt.github.io/_autosummary/torch_ttt.engine.it3_engine.IT3Engine.html)

> **For a quick tryout of the method, check the Colab notebooks below!**  


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

All experiments are available under the [`exps/`](./exps) directory. Each subfolder contains training and evaluation code, along with Colab notebooks where applicable. Some experiments are still being finalized and will be updated or added over time.

### MNIST Classification

We evaluate `IT³` on MNIST by training a model on clean digit images and testing it on inputs corrupted with Gaussian noise (σ = 0.5). The vanilla model suffers a sharp drop in accuracy due to the distribution shift. In contrast, `IT³` improves robustness by minimizing the idempotence error at inference time using only the current test input. This shows that `IT³` can adapt effectively even in simple, low-dimensional settings.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikitadurasov/ittt/blob/main/exps/mnist/mnist_ittt.ipynb)

---

### CIFAR-10 Classification

We evaluate `IT³` on CIFAR-10-C, a benchmark that introduces common corruptions such as blur, noise, and contrast shifts to the original CIFAR-10 images. The model is trained on clean images using the DLA architecture and tested on corrupted inputs (severity level 5). Compared to baselines like TTT and ActMAD, `IT³` achieves lower test error, and larger test-time batch sizes further improve performance.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikitadurasov/ittt/blob/main/exps/cifar/cifar_ittt.ipynb)

---

### Tabular Data (UCI)

We evaluate `IT³` on UCI regression datasets such as Boston Housing, where test inputs are corrupted by randomly zeroing out features (5–20%). A simple MLP is trained on clean data. As the level of corruption increases, performance of the vanilla model degrades significantly, while `IT³` maintains better accuracy by adapting to the OOD samples on-the-fly.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikitadurasov/ittt/blob/main/exps/tabular/vanilla_model.ipynb)

---

### Age Prediction (UTKFace)

We use the UTKFace dataset to regress a person's age from a face image. The model is trained on ages between 20 and 60, while faces outside this range are treated as out-of-distribution. `IT³` consistently reduces the mean absolute error compared to the non-adaptive model, confirming its effectiveness in real-valued regression tasks under covariate shift.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikitadurasov/ittt/blob/main/exps/age/eval_ittt.ipynb)

---

### Airfoil Regression (Physics)

We test `IT³` on aerodynamic prediction tasks, such as estimating lift-to-drag ratio from airfoil shapes using a Graph Neural Network. OOD samples include rare, high-performing shapes not seen during training. `IT³` is able to reduce the prediction error on these out-of-distribution inputs more effectively than ActMAD and vanilla models, demonstrating applicability to scientific and physics-based problems.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikitadurasov/ittt/blob/main/exps/airfoils/train_evaluate.ipynb)

---

### Road Segmentation

We train a DRU-Net on the RoadTracer dataset (urban roads) and evaluate on the Massachusetts Roads dataset (rural areas), introducing a domain shift. `IT³` is applied at test time and improves segmentation quality (measured by Correctness, Completeness, and IoU) over vanilla, TTT, and ActMAD baselines. Larger batch sizes during test-time adaptation further improve results.

Coming soon...

---

### ImageNet-C Classification

We evaluate `IT³` on ImageNet-C using a ResNet-18 and compare it to methods like TENT, MEMO, ETA, and ActMAD. Despite having no access to corrupted data during training, `IT³` consistently outperforms all baselines across 15 corruption types and multiple severity levels, with improvements scaling with larger batch sizes. This confirms the method’s scalability to large-scale classification under real-world corruptions.

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