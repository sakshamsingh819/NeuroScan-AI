
<div align="center">

# 🧠 NeuroScan-AI
### Brain Tumor Classification Using Deep Learning

**A computer vision project exploring CNNs, transfer learning, and fine-tuning for brain MRI image classification.**

<br/>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-6C5CE7?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed%20Experiments-2E8B57?style=for-the-badge)

<br/>

[**View Repository**](https://github.com/sujayv26/NeuroScan-AI)

</div>

---

## 📌 Overview

**NeuroScan-AI** is a deep learning project that classifies brain MRI images into four categories: Glioma, Meningioma, No Tumor, and Pituitary.

The project explores two approaches to image classification:

- **Custom Convolutional Neural Network (CNN):** A CNN developed for the target classification task.
- **EfficientNet-B0:** A pretrained model evaluated using transfer learning and further fine-tuning.

The objective is to implement, train, and evaluate different deep learning approaches, then compare their performance on the same classification problem.

> ⚠️ **Medical Disclaimer**
>
> NeuroScan-AI is an educational machine learning project. It is not a clinically validated diagnostic system and must not be used to diagnose, treat, or make medical decisions.

---

## 🎯 Project Objectives

- Build a custom CNN for brain MRI classification.
- Apply transfer learning using EfficientNet-B0.
- Fine-tune a pretrained model for the target dataset.
- Evaluate models using validation and test accuracy.
- Compare the performance of different model configurations.
- Practice an end-to-end deep learning workflow, from dataset preparation to model evaluation.

---

## 🗂️ Dataset

The project uses a brain tumor MRI image dataset containing four classes.

| Class | Description |
|:---|:---|
| `glioma` | Images labeled as glioma |
| `meningioma` | Images labeled as meningioma |
| `notumor` | Images labeled as no tumor |
| `pituitary` | Images labeled as pituitary |

### Dataset Distribution

| Dataset Split | Images |
|:---|---:|
| Training | 4,480 |
| Validation | 1,120 |
| Testing | 1,600 |
| **Total** | **7,200** |

The original training set contains 5,600 images. It was divided into training and validation subsets, while the separate test set contains 1,600 images.

### Image Preprocessing

The implemented data pipeline includes:

- Resizing images to `224 × 224`.
- Converting images to RGB.
- Normalizing images using ImageNet statistics for the model pipeline.
- Applying training-time augmentation, including random horizontal flips and rotation.

The dataset itself is **not included** in this repository.

---

## 🧠 Models & Experiments

### 1. Custom CNN — Baseline

A convolutional neural network developed for the four-class classification task.

The model learns image features through convolutional layers and uses a classification head to predict the target category.

**Purpose:** Establish a baseline using a custom-built architecture.

### 2. EfficientNet-B0 — Transfer Learning

EfficientNet-B0 is used as a pretrained feature extractor, with its classification head adapted to the four target classes.

**Purpose:** Evaluate the performance of pretrained image features on the brain MRI dataset.

### 3. EfficientNet-B0 — Fine-Tuning

The EfficientNet-B0 model is further trained on the target dataset by fine-tuning model parameters.

**Purpose:** Investigate whether adapting pretrained features to the target dataset improves classification performance.

---

## 📊 Results & Performance

The following table summarizes the experimental results.

| Model | Validation Accuracy | Test Accuracy |
|:---|---:|---:|
| Custom CNN | **94.46%** | **90.69%** |
| EfficientNet-B0 — Transfer Learning | 90.89% | 83.56% |
| EfficientNet-B0 — Fine-Tuned | 91.25% | 84.50% |

### Key Observations

- The custom CNN achieved **90.69% test accuracy** in the reported experiment.
- Fine-tuning EfficientNet-B0 improved its test accuracy by **0.94 percentage points**, from 83.56% to 84.50%.
- The custom CNN achieved higher test accuracy than the two EfficientNet-B0 configurations evaluated.

These are experimental results for this dataset and setup. Accuracy alone does not establish clinical reliability, and further evaluation would be required before drawing broader conclusions.

---

## 🏗️ Project Workflow

```text
Brain MRI Dataset
       │
       ▼
Dataset Inspection & Preparation
       │
       ▼
Image Preprocessing
       │
       ├──────────────────────┐
       ▼                      ▼
   Custom CNN           EfficientNet-B0
       │                 Transfer Learning
       │                      │
       │                      ▼
       │                   Fine-Tuning
       │                      │
       └──────────┬───────────┘
                  ▼
          Model Evaluation
                  │
                  ▼
       Validation & Test Results
```

---

## 📁 Repository Structure

```text
NeuroScan-AI/
│
├── Data/                         # Local dataset (not tracked)
│
├── models/
│   ├── efficientnet_b0_best.pth
│   └── efficientnet_b0_finetuned.pth
│
├── src/
│   ├── dataset.py                # Dataset loading and preprocessing
│   ├── dataset_check.py          # Dataset validation
│   ├── image_analysis.py         # Image analysis utilities
│   ├── view_images.py            # Image visualization
│   ├── model.py                  # Custom CNN architecture
│   ├── train.py                  # Custom CNN training
│   ├── evaluate.py               # Custom CNN evaluation
│   ├── transfer_model.py         # Transfer learning model
│   ├── train_transfer.py         # EfficientNet training
│   ├── finetune.py               # EfficientNet fine-tuning
│   └── evaluate_transfer.py      # EfficientNet evaluation
│
├── .gitignore
└── README.md
```

**Note:** The custom CNN checkpoint is not included in the repository. The EfficientNet-B0 checkpoints are provided under `models/`.

---

## ⚙️ Getting Started

### Prerequisites

- Python
- Git
- pip
- A compatible PyTorch environment

### 1. Clone the repository

```bash
git clone https://github.com/sujayv26/NeuroScan-AI.git
cd NeuroScan-AI
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install PyTorch, Torchvision, and the other packages required by the project.

If a `requirements.txt` file is present:

```powershell
python -m pip install -r requirements.txt
```

Otherwise, install the dependencies according to the imports and environment used by the project.

### 4. Prepare the dataset

Place the dataset in the expected local `Data/` directory.

Ensure the folder names and paths match the configuration in `src/dataset.py`.

---

## ▶️ Running the Experiments

Run the following commands from the repository root.

### Inspect the dataset

```powershell
python src/dataset_check.py
```

### Train the custom CNN

```powershell
python src/train.py
```

### Evaluate the custom CNN

```powershell
python src/evaluate.py
```

### Train EfficientNet-B0

```powershell
python src/train_transfer.py
```

### Fine-tune EfficientNet-B0

```powershell
python src/finetune.py
```

### Evaluate EfficientNet-B0

```powershell
python src/evaluate_transfer.py
```

**Important:** Training and evaluation require the dataset and the relevant model checkpoints. Check the paths configured in each script before running it. The custom CNN checkpoint must be generated by training or obtained separately.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|:---|:---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| Torchvision | Computer vision models and transforms |
| CNN | Custom image classification architecture |
| EfficientNet-B0 | Transfer learning and fine-tuning |
| Git & GitHub | Version control and project hosting |
| Visual Studio Code | Development environment |

---

## 📚 Key Learnings

Through this project, I explored:

- Image dataset inspection and preprocessing.
- CNN architecture implementation.
- Multi-class image classification.
- Transfer learning with pretrained models.
- Fine-tuning deep learning models.
- Validation and test-set evaluation.
- Model checkpoint management.
- Version control and GitHub project organization.

---

## 🚀 Future Improvements

Potential next steps for the project include:

- [ ] Add a confusion matrix and per-class precision, recall, and F1-score.
- [ ] Visualize training and validation loss and accuracy.
- [ ] Add Grad-CAM visualizations for model interpretability.
- [ ] Build a simple web interface for image classification.
- [ ] Add a reproducible dependency file and experiment configuration.
- [ ] Evaluate robustness across different datasets and imaging conditions.

---

## 👨‍💻 Author

**Sujay V**

B.Tech — Computer Science and Engineering  
Specialization: AI-Driven DevOps

[GitHub Profile](https://github.com/sujayv26) · [Project Repository](https://github.com/sujayv26/NeuroScan-AI)

---

## 📄 License

No project license has been specified yet. Before redistributing the code, model weights, or dataset, review the applicable licenses and usage terms.

---

<div align="center">

**NeuroScan-AI**  
*Exploring Deep Learning for Medical Image Classification*

</div>