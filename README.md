# Zero To Hero Tutorial on a Deep Learning Classification Task

This repository tries to introduce the different stages that someone needs to follow for classifying data based on their
complexity. 
More information can be found in the interactive 
[Google Colab Notebooks](https://colab.research.google.com/drive/1P6OHYSrYYqQ7qgBVtTphOvkQztYClv13?usp=sharing).

## Description

In Deep Learning it is required multiple times to distinguish data between them. The task that tackles this challenge is
the classification task. In this repository we are going to present how we can classify the following datasets with deep
learning models:

Datasets:

1. [Gaussian-blobs]
2. fashion items ([fashionMNIST])
3. relations among authors ([ogb-collab])

Models:

1. MultiLayer-Perceptron (MLP)
2. Convolutional Neural Networks (CNNs)
3. Graph Convolutional Networks (GCNs)

We can combine the aforementioned models to build different architectures which will help us to solve the classification
task for each dataset.

The implementations of the different stages are based on the following frameworks:

- [PyTorch]
- [PyTorch-Lightning]
- [Lightning-Bolts]
- [DGL] (Deep Graph Library)
- [OGB] (Open Graph Benchmark)

## Installation

In order to set up the necessary virtual environment:

1. review and uncomment / comment based on GPU availability and CUDA version of your machine what you need
   in `requirements.txt`
   and create a virtual environment `.venv`:
   ```bash
   python3 -m venv .venv
   ```
2. activate the new environment with:
   ```bash
   source .venv/bin/activate  # Mac & Linux users
   .venv\Scripts\activate  # Windows users
   ```
3. update `pip` package:
   ```bash
   python -m pip install --upgrade pip
   ```
4. install `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
5. [PyTorch-installation] based on GPU availability
6. [DGL-installation] based on GPU availability
7. install `zero_to_hero` package:
   ```bash
   pip install -e .
   ```

> **_NOTE:_**  The virtual environment will have `zero_to_hero` installed in editable mode.
> Some changes, e.g. in `setup.cfg`, might require you to run `pip install -e .` again.

Optional and needed only once after `git clone https://github.com/Deligiorgis/zero_to_hero.git`:

8. install several [pre-commit] git hooks with:
   ```bash
   pre-commit install
   # You might also want to run `pre-commit autoupdate`
   ```
   and checkout the configuration under `.pre-commit-config.yaml`. The `-n, --no-verify` flag of `git commit` can be
   used to deactivate pre-commit hooks temporarily.

## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data                    <- The contents have been ignored
│   ├── FashionMNIST        <- FashionMNIST data will be downloaded by default here.
│   ├── ogbl_collab         <- OGBL-Collab data will be downloaded by default here.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── pyproject.toml          <- Build system configuration. Do not change!
├── scripts                 <- Analysis and production scripts which import the
│                              actual Python package, e.g. train_model.py.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Use `pip install -e .` to install for development or
|                              or create a distribution with `tox -e build`.
├── src
│   └── zero_to_hero        <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `py.test`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

## How to run the classification task for:

Assuming that you are using the aforementioned generated virtual environment.

### Gaussian Blobs

To run the script that classifies the Gaussian Blobs first you'll need to choose how many dimensions you want for the
blobs and which architecture to use
(further information or examples can be found in
`configs/blobs.yml`). Then run the following command:

```bash
python scripts/main_classify_blobs.py
```

### Fashion-MNIST

To run the script that classifies the FashionMNIST dataset first you'll need to choose which architecture to use
(further information or examples can be found in
`configs/fashion_mnist.yml`). Then run the following command:

```bash
python scripts/main_classify_fashion_mnist.py
```

### OGBL-Collab

To run the script that predicts the links (co-authors)
first you'll need to choose which architecture to use
(further information or examples can be found in
`configs/collab.yml`). Then run the following command:

```bash
python scripts/main_link_prediction_collab.py
```

### Tensorboard

To monitor the progress of the model's training you can use TensorBoard by running the following command:

```bash
tensorboard --logdir=tensorboard_logs
```

## References & Acknowledgments

Papers:

- [Revisiting Graph Neural Networks for Link Prediction]
- [Link Prediction Based on Graph Neural Networks]
- [An End-to-End Deep Learning Architecture for Graph Classification]

GitHub:

- https://github.com/facebookresearch/SEAL_OGB
- https://github.com/dmlc/dgl/tree/master/examples/pytorch/seal
- https://github.com/muhanzhang/DGCNN
- https://github.com/muhanzhang/pytorch_DGCNN

<!-- pyscaffold-notes -->

## Note

This project has been set up using [PyScaffold] 4.0.2 and the [dsproject extension] 0.6.1.

[pre-commit]: https://pre-commit.com/
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
[fashionMNIST]: https://github.com/zalandoresearch/fashion-mnist
[ogb-collab]: https://ogb.stanford.edu/docs/linkprop/#ogbl-collab
[Gaussian-blobs]: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
[PyTorch]: https://pytorch.org/
[OGB]: https://ogb.stanford.edu/
[DGL]: https://www.dgl.ai/
[DGL-installation]: https://www.dgl.ai/pages/start.html
[PyTorch-installation]: https://pytorch.org/
[PyTorch-Lightning]: https://www.pytorchlightning.ai/
[Lightning-Bolts]: https://lightning-bolts.readthedocs.io/en/latest/
[Link Prediction Based on Graph Neural Networks]: https://arxiv.org/abs/1802.09691
[Revisiting Graph Neural Networks for Link Prediction]: https://arxiv.org/abs/2010.16103
[An End-to-End Deep Learning Architecture for Graph Classification]: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/17146
