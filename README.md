
# T-GRAASP

**T-GRAASP: Reconstructing Spatial Context in Highly Heterogeneous Tissues**

---

## Overview

**T-GRAASP** is a computational framework designed for reconstructing the spatial organization of cells and analyzing cellular interactions in highly heterogeneous tissues, based on spatial transcriptomics data. T-GRAASP leverages graph-based modeling to accurately infer spatial adjacency, identify spatial domains, and dissect gene interaction networks across various tissue types.

---

## Installation

### 1. Conda Installation (Recommended)

Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed.

#### **A. Clone the repository**
```bash
git clone https://github.com/caofy0307/T-GRAASP.git
cd T-GRAASP
```

#### **B. Create environment from environment.yml**
If an `environment.yml` file is provided (recommended for full reproducibility):

```bash
conda env create -f environment.yml
conda activate t-graasp
```

---

## Usage

This project provides detailed tutorial notebooks to help you get started quickly.  
After installing the environment, simply open the following Jupyter notebooks in the repository:

- **[Pretrain.ipynb](https://github.com/caofy0307/T-GRAASP/tree/main/tutorials/Pretrain.ipynb)**  
  *This notebook demonstrates the pretraining workflow of the T-GRAASP model. Use it to learn how to prepare your data and pretrain the model for spatial transcriptomics analysis.*

- **[Finetuned.ipynb](https://github.com/caofy0307/T-GRAASP/tree/main/tutorials/Finetuned.ipynb)**  
  *This notebook shows how to fine-tune the pretrained T-GRAASP model on your own datasets and analyze results, including visualization and downstream analyses.*

- **[Downstream.ipynb](https://github.com/caofy0307/T-GRAASP/tree/main/tutorials/Downstream.ipynb)**  
  *This notebook provides downstream biological analysis and rich visualization examples, helping you to interpret and explore the results from T-GRAASP in a biological context.*

#### How to run the tutorials:
1. **Activate the Conda environment:**
   ```bash
   conda activate t-graasp
   ```
2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
3. **Open `Pretrain.ipynb`, `Finetuned.ipynb`, and `Downstream.ipynb` in your browser and follow the step-by-step instructions.**  
   These notebooks are located in the root or `notebooks/` directory of this repository.

> The notebooks include comments and code cells that are ready to run.  
> You can adapt them for your own data or use the provided example datasets.

---

## Contact

For questions, bug reports, or collaborations, please open an issue.
