# COMP5511-assign2

This repository contains my implementation for **COMP5511 Assignment 2**, including:

1. **Task I – Linear Regression for Housing Price Prediction**
2. **Task II – Denoising Autoencoder on Fashion-MNIST**
3. **Task III – Using a Large Language Model (LLM) to Assist Neural Network Design**
4. **Task IV – Reinforcement Learning (Q-learning) on FrozenLake-v1**

All code is written in Python and tested with **Python 3.9** in a Conda environment.

---

## 1. Repository Structure

```text
Assign2/
├── FASHION-MNIST/
│   ├── task2_fashion_denoise.py
│   ├── task2_denoising_examples.png      # Noisy / Denoised / Clean images
│   ├── task2_learning_curve.png          # Train / Val MSE curve
│   ├── train_clean.csv
│   ├── train_noisy.csv
│   ├── test_clean.csv
│   └── test_noisy.csv
│
├── LINEAR REGRESSION/
│   ├── ParisHousing.csv
│   └── task1_paris_housing.py
│
├── REINFORCEMENT LEARNING FOR GAME/
│   ├── frozenlake_q_learning.py
│   ├── frozenlake_learning_curve.png     # Success rate vs. episode
│   ├── policy_det.gif                    # Greedy policy, is_slippery=False
│   └── policy_sto.gif                    # Greedy policy, is_slippery=True
│
└── report/
    └── Assignment2_Report.docx           # Final written report (template filled in)
````

> **Note:** The exact folder names may differ slightly depending on where you place the report file.

---

## 2. Environment & Dependencies

Create a Conda environment (recommended) and install the required packages:

```bash
conda create -n comp5511 python=3.9
conda activate comp5511
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
pip install torch torchvision
pip install gymnasium imageio pygame
```

If Gymnasium complains about missing extras for FrozenLake, install:

```bash
pip install "gymnasium[toy-text]"
```

---

## 3. Task I – Linear Regression on ParisHousing

**Script:** `LINEAR REGRESSION/task1_paris_housing.py`

### Description

* Loads `ParisHousing.csv`.
* Performs train/test split and standardization of features.
* Trains a **baseline Linear Regression model** on the full dataset.
* Optionally trains segmented models:

  * Per value of `numPrevOwners`
  * Per K-means cluster
* Evaluates models using **MAE**, **RMSE**, and **R²** on the test set.

### Run

```bash
cd "LINEAR REGRESSION"
python task1_paris_housing.py
```

The script prints evaluation metrics to the console; you can copy them into the report.

---

## 4. Task II – Denoising Autoencoder on Fashion-MNIST

**Script:** `FASHION-MNIST/task2_fashion_denoise.py`

### Description

* Loads paired noisy/clean images from the four CSV files.

* Builds a **convolutional autoencoder**:

  * Encoder: Conv → ReLU → MaxPool → Conv → ReLU → MaxPool
  * Decoder: ConvTranspose → ReLU → ConvTranspose → Sigmoid

* Trains with **MSE loss** using Adam optimizer.

* Saves:

  * `task2_learning_curve.png`: training & validation MSE vs. epoch
  * `task2_denoising_examples.png`: grid of noisy / denoised / clean images

### Run

```bash
cd "FASHION-MNIST"
python task2_fashion_denoise.py
```

Outputs (figures) are saved in the same folder and referenced in the report.

---

## 5. Task III – LLM for Neural Network Design

Task III is mainly **conceptual** and documented in the written report:

* I used ChatGPT as the LLM to:

  * Summarize the history and applications of large language models.
  * Suggest improvements to the Task II autoencoder architecture and training scheme.
  * Help analyze and discuss the advantages/limitations of using LLMs as design assistants.

There is no separate executable script for Task III; the discussion is integrated in the report.

---

## 6. Task IV – Reinforcement Learning on FrozenLake-v1

**Script:** `REINFORCEMENT LEARNING FOR GAME/frozenlake_q_learning.py`

### Description

* Creates the **8×8 FrozenLake-v1** environment in two modes:

  * Deterministic: `is_slippery=False`
  * Stochastic: `is_slippery=True`

* Trains a **tabular Q-learning agent** for each environment:

  * α = 0.8, γ = 0.99
  * ε-greedy exploration (ε starts at 1.0, decays to 0.05)
  * 20 000 episodes

* Records success (goal reached) for each episode.

* Computes moving average success rate over the last 100 episodes.

* Saves:

  * `frozenlake_learning_curve.png`: success rate curves for both environments
  * `policy_det.gif`: greedy policy rollout for deterministic environment
  * `policy_sto.gif`: greedy policy rollout for stochastic environment

### Run

```bash
cd "REINFORCEMENT LEARNING FOR GAME"
python frozenlake_q_learning.py
```

After training, check the folder for the learning curve image and GIFs and embed them into the report.

---

## 7. Reproducibility Notes

* Random seeds (NumPy / PyTorch) are set inside the scripts where appropriate, but due to library and GPU differences, small variations can occur.
* All experiments were run on CPU; no GPU-specific code is required.
* If you encounter path issues, run the scripts from the project root and adjust relative paths accordingly.

---

## 8. Contact

If you have any questions about the code or report structure, please contact:

* **Name:** (fill in your name here)
* **Email:** (your PolyU / university email here)

```

你可以把上面整段复制到 `README.md` 里，然后把名字和邮箱改成你自己的，就齐活了。
::contentReference[oaicite:0]{index=0}
```
