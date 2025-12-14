# OGBN-Arxiv GCN Hyperparameter Search

## Overview

This project is dedicated to training, evaluating, and optimizing Graph Convolutional Network (GCN) models for the node classification task on the `ogbn-arxiv` dataset. It provides a structured framework for performing hyperparameter tuning (grid search) to discover the most effective model architectures and training configurations.

The repository contains two main experiments: a baseline GCN implementation and a more advanced version that incorporates modern techniques like Jumping Knowledge and Node Encoders.

## Features

- **Model**: Implements Graph Convolutional Networks using **PyTorch** and **PyTorch Geometric**.
- **Dataset**: Utilizes the `ogbn-arxiv` computer science citation graph from the **Open Graph Benchmark (OGB)**.
- **Hyperparameter Tuning**: A systematic grid search methodology to explore various model configurations.
- **Early Stopping**: Training automatically halts if validation accuracy ceases to improve, saving the best-performing model state to prevent overfitting and reduce training time.
- **Comprehensive Logging**: Experiment results are saved in multiple formats for easy analysis:
    - A summary CSV file for high-level results.
    - Detailed per-epoch JSON logs for fine-grained analysis.
    - The best model weights (`.pth`) for each run.
- **Advanced Architectures**: The `v02` script explores modern GCN improvements like Residual Connections, Jumping Knowledge, and separate Node Encoders.

## Repository Structure

```
.
├── grid_search_arxiv_v01.py    # Script for the baseline GCN experiment
├── grid_search_arxiv_v02.py    # Script for the advanced GCN experiment (with JK, etc.)
├── requirements.txt            # Python dependencies
├── notebooks/                  # Jupyter notebooks for data exploration and results analysis
├── dataset/                    # Default directory for the downloaded ogbn-arxiv dataset
└── outputs/                    # Default directory for all generated artifacts
    ├── experiments_log_v*.csv  # Summary logs of all experiments
    ├── logs/                   # Detailed JSON history for each run
    └── models/                 # Saved .pth model checkpoints
```

## Methodology

The project follows an iterative approach to finding the best model.

### `v01`: Baseline GCN

This version tests fundamental hyperparameters to establish a strong baseline. The search space includes:
- `num_layers`: The depth of the GCN.
- `hidden_dim`: The dimensionality of the node embeddings.
- `norm_type`: The type of normalization to apply (e.g., `batch`, `layer`, `graph`).
- `use_residual`: Whether to include residual (skip) connections.

### `v02`: Advanced GCN

This iteration builds upon the baseline by incorporating more sophisticated architectural techniques to improve performance:
- **Node Encoder**: A `Linear` layer that projects the input features into the GCN's hidden dimension *before* the first graph convolution.
- **Jumping Knowledge (JK)**: Implemented in `'cat'` mode, this technique aggregates the output from all intermediate GCN layers, allowing the model to learn from different neighborhood depths simultaneously. This helps combat the "over-smoothing" problem in deep GCNs.
- **Refined Search Space**: The hyperparameter grid is updated, fixing previously discovered optimal settings (like `use_residual`) and focusing the search on new parameters like `use_jk`.

## How to Run

### 1. Prerequisites
- Python 3.8+
- PyTorch (see official site for installation instructions: https://pytorch.org/)
- CUDA (recommended for GPU acceleration)

### 2. Installation
Clone the repository and install the required Python packages.
```bash
git clone https://github.com/your-username/ogbn-arxiv-gcn.git
cd ogbn-arxiv-gcn
pip install -r requirements.txt
```

### 3. Execution
Run either of the grid search scripts. The script will automatically handle data downloading, preprocessing, training, evaluation, and logging.

```bash
# To run the baseline experiment
python grid_search_arxiv_v01.py

# To run the advanced experiment
python grid_search_arxiv_v02.py
```
The scripts will print progress to the console and save all results in the `outputs/` directory.

## Understanding the Outputs

All artifacts are saved in the `outputs/` directory:

- **CSV Log (`/outputs/experiments_log_v*.csv`)**: A summary of each experiment run, including the hyperparameter configuration, final accuracies, training duration, and file paths to the detailed logs and model.
- **JSON History (`/outputs/logs/`)**: A detailed, epoch-by-epoch record of metrics (e.g., `train_loss`, `valid_acc`, `test_acc`) for each run.
- **Model Checkpoints (`/outputs/models/`)**: The PyTorch model state dictionary (`.pth` file) corresponding to the epoch with the best validation accuracy for each experiment. This file can be used to load the trained model for inference or further analysis.
