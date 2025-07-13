# Ceb

## 🔥 Ceb Overview

![overview](figures/overview.png)


## 🚀 Installation

This project requires matlab package

```
conda create -n Ceb python=3.10
conda activate Ceb

conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia

conda install -c conda-forge scikit-image pillow numpy scikit-learn cvxpy imagecodecs

conda install -c conda-forge protobuf abseil-cpp opencv

pip install cvxopt matplotlib

pip install anytree
```

## 📊 Dataset Preparation

This project applies a deep learning model (e.g., U-Net) to generate probability maps for both the training and test sets. These probability maps represent the per-pixel likelihood of cell instances in raw images.

📌  Training Set: Probability maps are generated using a five-fold cross-validation approach. Each fold's model produces predictions on its held-out subset.

📌  Test Set: Probability maps are generated using a model trained on the entire training set.


Toy examples are provided in the [`/examples/`](examples/) directory.

## 🔬 Experimental Pipeline

### Generate boundary signatures and corresponding boundary labels

```
sh Scrips/generate_cell_signatures_SIM.sh
```

### Train the boundary classifier and apply boundary classifier to get cell instance segmentation results

```
sh Scripts/train_classifier_SIM.sh
```

### For the video datasets, apply a matching step to further improve cell segmentation results

```
sh Scripts/SIM_temporal.sh
```



