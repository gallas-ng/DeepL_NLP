# Sentiment Classification with DistilBERT

## Project Overview

This project explores sentiment classification of movie reviews using a pretrained **DistilBERT** model and the IMDb dataset.

The main objective is to compare two transfer-learning strategies:

1. **Fine-tuning:** the pretrained DistilBERT encoder and classification head are trained together.
2. **Frozen encoder:** the pretrained DistilBERT encoder is kept frozen and only the classification head is trained.

The experiment investigates the following question:

> **How does fine-tuning a pretrained DistilBERT model affect sentiment classification performance compared with freezing the pretrained encoder?**

---

## Dataset

The project uses the **IMDb movie review dataset** from Hugging Face Datasets.

The original dataset contains:

* 25,000 training reviews
* 25,000 test reviews
* Binary sentiment labels: positive / negative

For this experiment, a controlled subset was used:

| Split      | Samples |
| ---------- | ------: |
| Training   |   4,000 |
| Validation |   1,000 |
| Test       |   1,000 |

---

## Model

The pretrained model used is:

`distilbert-base-uncased`

DistilBERT is a smaller and faster version of BERT while retaining strong language representation capabilities.

Two configurations were evaluated:

| Strategy       | Trainable components                     |
| -------------- | ---------------------------------------- |
| Fine-tuning    | DistilBERT encoder + classification head |
| Frozen encoder | Classification head only                 |

---

## Experimental Setup

| Parameter               | Value                     |
| ----------------------- | ------------------------- |
| Model                   | `distilbert-base-uncased` |
| Maximum sequence length | 256                       |
| Batch size              | 16                        |
| Learning rate           | 5e-5                      |
| Epochs                  | 2                         |
| Weight decay            | 0.01                      |
| Random seed             | 42                        |
| Hardware                | NVIDIA Tesla T4           |
| Precision               | FP16                      |

The same dataset, preprocessing pipeline, tokenizer, hyperparameters, and evaluation metrics were used for both strategies to ensure a controlled comparison.

---

## Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Evaluation loss

### Results

| Strategy       |  Accuracy | Precision |    Recall |        F1 |
| -------------- | --------: | --------: | --------: | --------: |
| Fine-tuning    | **87.8%** | **88.0%** |     86.9% | **87.4%** |
| Frozen encoder |     79.3% |     77.7% | **80.7%** |     79.2% |

Fine-tuning achieved higher accuracy, precision, and F1-score, while the frozen encoder achieved slightly higher recall.

---

## Project Structure

```text
.
├── notebook/
│   └── sentiment_classification_distilbert.ipynb
├── requirements.txt
├── README.md
└── report/
    └── project_report.pdf
```

The notebook contains the complete experimental pipeline, including:

* Dataset loading
* Exploratory data analysis
* Data preprocessing
* Tokenization
* Model configuration
* Fine-tuning
* Frozen-encoder training
* Model evaluation
* Performance comparison
* Confusion matrices and visualizations

---

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

The project can be run using Jupyter Notebook or Google Colab.

---

## Reproducibility

The experiments use a fixed random seed:

```python
SEED = 42
```

The main configuration parameters are defined explicitly in the notebook. This allows the preprocessing, training, and evaluation pipeline to be reproduced under the same experimental setup.

---

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face Evaluate
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

## Conclusion

The experiment shows that fine-tuning the pretrained DistilBERT encoder provides a substantial performance improvement for IMDb sentiment classification compared with training only a classification head on frozen representations.
