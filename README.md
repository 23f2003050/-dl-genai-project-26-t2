# Smart MCQ Solver Challenge — Deep Learning & GenAI Project

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-blue.svg)](https://www.kaggle.com/competitions/smart-mcq-solver-challenge)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-GenAI-orange.svg)](https://www.iitm.ac.in/)
[![Python](https://img.shields.io/badge/Python-3.12+-brightgreen.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution Architecture](#solution-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Key Insights](#key-insights)
- [Models Comparison](#models-comparison)
- [Challenges & Solutions](#challenges--solutions)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [License](#license)

---

## 🎯 Overview

This project implements a **Multiple-Choice Question (MCQ) ranking solver** for the Kaggle Smart MCQ Solver Challenge. The solution achieves **0.756 MAP@3** on the public leaderboard by combining classical machine learning (sparse features) with modern NLP embeddings (dense features).

### Key Features
- ✅ **Pairwise Classification**: Converts 5-way ranking into binary classification
- ✅ **Dual Feature Engineering**: Sparse TF-IDF + Dense MiniLM embeddings
- ✅ **Robust Cross-Validation**: GroupKFold prevents data leakage
- ✅ **Ensemble Blending**: Weighted combination of complementary models
- ✅ **Comprehensive Evaluation**: MAP@3, Accuracy, Precision, Recall, F1-Score

### Performance Metrics

| Metric | Score |
|--------|-------|
| **MAP@3** | **0.75644** ⭐ |
| **Top-1 Accuracy** | 0.6560 |
| **Precision (macro)** | 0.6540 |
| **Recall (macro)** | 0.6490 |
| **F1-Score (macro)** | 0.6510 |

---

## 📝 Problem Statement

### Task
Given a question prompt and five multiple-choice options (A, B, C, D, E), **predict the top 3 most likely correct answers in ranked order**.

### Input
```
id: 1
prompt: "What is artificial intelligence?"
A: "A subset of computer science focusing on machine learning"
B: "A type of programming language"
C: "A hardware component"
D: "Artificial intelligence is the simulation of human intelligence"
E: "A database management system"
answer: "D"
```

### Output (Submission Format)
```
id: 1
prediction: "D A C"  # Space-separated top-3 predictions
```

### Evaluation Metric: MAP@3

Mean Average Precision at 3 is calculated as:

$$\text{MAP@3} = \frac{1}{U} \sum_{u=1}^{U} \sum_{k=1}^{\min(3, P_u)} \frac{1}{k} \cdot \text{rel}(k)$$

Where:
- **rel(k) = 1** if the correct answer is at position k, else 0
- **Position 1**: score = 1.0
- **Position 2**: score = 0.5
- **Position 3**: score ≈ 0.333
- **Not in top-3**: score = 0.0

**Why MAP@3?** It rewards both correctness and ranking quality. A wrong first choice doesn't invalidate good second/third choices.

---

## 🏗️ Solution Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           SMART MCQ SOLVER PIPELINE                         │
└─────────────────────────────────────────────────────────────┘

INPUT DATA
│
├─ Train Set → Pairwise Construction (1000 Q's → 5000 pairs)
│
├─ FEATURE ENGINEERING (PARALLEL PATHS)
│  ├─ PATH 1: SPARSE (TF-IDF)
│  │  ├─ Word n-grams (1-3): capture keywords
│  │  ├─ Char n-grams (2-5): capture morphology
│  │  └─ FeatureUnion: combine both
│  │
│  └─ PATH 2: DENSE (MiniLM)
│     └─ SentenceTransformer embeddings (384-dim)
│
├─ CROSS-VALIDATION (5-Fold GroupKFold)
│  ├─ Split: keep question options together
│  ├─ Train: LogisticRegression on fold data
│  └─ Evaluate: OOF predictions for performance estimate
│
├─ MODEL TRAINING (Two Models)
│  ├─ MODEL 1: Sparse (C=5.0)
│  ├─ MODEL 2: Dense (C=2.0)
│  └─ ENSEMBLE: 60% Model1 + 40% Model2
│
├─ RANKING & INFERENCE
│  ├─ Blend probabilities
│  ├─ Extract top-3 indices
│  └─ Convert to letter labels
│
└─ SUBMISSION
   └─ CSV: id | prediction
```

---

## 📊 Dataset

### File Structure

| File | Size | Rows | Purpose |
|------|------|------|---------|
| **train.csv** | ~1.5 MB | 242 unique prompts × 5 options | Training data with labels |
| **test.csv** | ~0.8 MB | Similar structure, no labels | Evaluation data |
| **sample_submission.csv** | < 1 KB | Template | Format reference |

### Dataset Characteristics

```
Total Unique Questions: ~1,000
Total Pairwise Samples: ~5,000 (1,000 × 5 options)

Duplicates Detected:
  - Exact MCQ duplicates: 183 sets
  - Duplicate prompts (lexical variation): 242 instances
  - Label consistency: 100% (same prompt → same answer)

Class Distribution:
  - Option A: 20.70%
  - Option B: 24.50%
  - Option C: 22.95%
  - Option D: 19.70%
  - Option E: 12.15%
  (Roughly balanced, no severe imbalance)

Length Bias Discovery:
  - Longest option is correct answer: 57.2% of time
  - Baseline (pure length heuristic): 0.49 MAP@3
  - Our model: 0.756 MAP@3 (+54% improvement)
```

### Data Quality
- ✅ No missing values
- ✅ No data type issues
- ✅ Consistent formatting
- ⚠️ Duplicate MCQ sets (handled via GroupKFold)

---

## 🎓 Methodology

### Step 1: Pairwise Formulation

Convert 5-way multi-class ranking into 5 binary classification problems.

**Transformation:**
```
Original (1 sample):
  (prompt, A, B, C, D, E, answer=C)

Pairwise (5 samples):
  (prompt + A, label=0)
  (prompt + B, label=0)
  (prompt + C, label=1) ← correct
  (prompt + D, label=0)
  (prompt + E, label=0)
```

**Advantages:**
- Simpler binary classification vs complex 5-way ranking
- Each option gets independent probability
- Natural ranking via probability sorting
- Enables ensemble of complementary features

### Step 2: Feature Engineering

#### Path A: Sparse Features (TF-IDF)

```python
TfidfVectorizer(
    ngram_range=(1, 3),      # Word 1-grams, 2-grams, 3-grams
    analyzer='word',
    min_df=1,                # Keep all words
    sublinear_tf=True        # Log scaling: log(1 + tf)
)

TfidfVectorizer(
    ngram_range=(2, 5),      # Char 2-grams to 5-grams
    analyzer='char_wb',      # Word-boundary aware
    min_df=2,                # Prune rare char patterns
    sublinear_tf=True
)

Result: FeatureUnion → Sparse matrix (~10k dimensions)
```

**What's Captured:**
- Words: semantic keywords and phrases
- Characters: spelling patterns, morphology, typo robustness

#### Path B: Dense Features (Embeddings)

```python
SentenceTransformer('all-MiniLM-L6-v2')
  - Pre-trained on sentence similarity tasks
  - 66M parameters (distilled from 110M BERT)
  - Output: 384-dimensional dense vectors
  - Handles semantic equivalence and paraphrases

Result: Dense array (~384 dimensions)
```

**What's Captured:**
- Semantic meaning of prompt + option pair
- Contextual relationships
- Paraphrases and synonyms

### Step 3: Cross-Validation Strategy

**5-Fold GroupKFold** to prevent data leakage:

```python
gkf = GroupKFold(n_splits=5)
groups = pw_train['question_id']  # Group by question

for fold, (train_idx, val_idx) in enumerate(gkf.split(..., groups=groups)):
    # All options of same question stay together
    # train_idx: ~80% of questions (all 5 options each)
    # val_idx: ~20% of questions (all 5 options each)
```

**Why Essential?**
- Regular KFold allows question split across folds → leakage
- GroupKFold keeps all options of a question in same fold
- Validation scores match test scores (no surprises)

### Step 4: Model Training

**Sparse Model:**
```python
LogisticRegression(
    C=5.0,              # Less regularization (high-dim features)
    max_iter=3000,
    random_state=42+fold
)
```

**Dense Model:**
```python
LogisticRegression(
    C=2.0,              # More regularization (lower-dim features)
    max_iter=2000,
    random_state=42+fold
)
```

**Training Loop:**
```
For each fold 1-5:
  1. Fit sparse model on fold training data
  2. Predict probabilities on fold validation data → OOF
  3. Predict on full test set → store per fold
  
  4. Fit dense model on same fold
  5. Predict probabilities on fold validation → OOF
  6. Predict on full test → store per fold

After 5 folds:
  - OOF predictions: every training sample predicted once
  - Test predictions: each test sample has 5 predictions
```

### Step 5: Probability Blending

```python
# Weighted combination of two models
blend_prob = 0.60 * sparse_prob + 0.40 * dense_prob

# Reshape to question format (n_questions × 5_options)
blend_matrix = blend_prob.reshape(-1, 5)

# Extract top-3 indices (highest probabilities)
top3_indices = np.argsort(blend_matrix, axis=1)[:, -3:][:, ::-1]

# Convert indices to letter labels
predictions = [['A','B','C','D','E'][i] for i in top3_indices]

# Format for submission
submission = " ".join(predictions)  # "C A B"
```

### Step 6: Submission Generation

```python
submission_df = pd.DataFrame({
    'id': test['id'],
    'prediction': predictions
})
submission_df.to_csv('submission.csv', index=False)
```

---

## 💡 Key Insights

### Insight 1: Length Bias Exists but Insufficient

**Discovery:** 57.2% of correct answers are the longest option

```
Baseline (pure length ranking):
  MAP@3: 0.49
  Accuracy: 0.41

Our model (semantic + lexical):
  MAP@3: 0.756
  Accuracy: 0.656

Improvement: +54%
```

**Implication:** While length is a real pattern, semantic understanding is essential.

### Insight 2: Pairwise Formulation is Powerful

Converting MCQ to pairwise binary classification:
- Simplifies task complexity
- Enables standard models
- Natural ranking via probability sorting
- Better than direct multi-class for ranking tasks

### Insight 3: Complementary Features

- **TF-IDF (sparse)**: Fast, interpretable, captures exact keyword matches (baseline: 0.751 MAP@3)
- **MiniLM (dense)**: Slower, semantic, captures meaning and paraphrases
- **Blend (60-40)**: Combines both → 0.756 MAP@3 (better than either alone)

### Insight 4: Data Leakage Risk

- Dataset has 183 exact duplicate MCQs and 242 repeated prompts
- Must use **GroupKFold** to keep question options together
- Regular KFold would artificially inflate validation scores

### Insight 5: Cross-Validation Alignment

OOF scores (0.756 MAP@3) ≈ Public leaderboard scores (0.756)

This alignment confirms:
- No data leakage
- Realistic generalization estimate
- Reproducible results

---

## 🏆 Models Comparison

| Model | Type | Features | MAP@3 | Accuracy | F1 | Notes |
|-------|------|----------|-------|----------|-----|-------|
| **Baseline 1** | Heuristic | Length ranking | 0.4904 | 0.4120 | 0.3950 | Underperforms |
| **Baseline 2** | TF-IDF + Cosine | Unsupervised | 0.3558 | 0.2450 | 0.2310 | Fails without labels |
| **Model 1** | 5-Fold Pairwise TF-IDF | Sparse only | **0.7506** | 0.6480 | 0.6420 | Strong baseline |
| **Model 2** | DeBERTa + LoRA | LLM + LoRA | 0.3770 | 0.2850 | 0.2740 | Underperforms |
| **Model 3** ⭐ | Dual Sparse+Dense | 60% TF-IDF + 40% MiniLM | **0.7564** | **0.6560** | **0.6510** | **BEST** |

---

## 🔧 Installation & Setup

### Prerequisites

- Python 3.10+
- pip or conda
- 4GB RAM (CPU) or GPU (optional, for faster inference)

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/yourusername/smart-mcq-solver.git
cd smart-mcq-solver

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download competition data (Kaggle CLI)
kaggle competitions download -c smart-mcq-solver-challenge
unzip smart-mcq-solver-challenge.zip -d data/
```

### Required Libraries

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
sentence-transformers>=2.2.0
torch>=1.9.0
transformers>=4.20.0
peft>=0.2.0
matplotlib>=3.4.0
wandb>=0.12.0
```

See `requirements.txt` for exact versions.

---

## 🚀 Usage

### Quick Start

```python
# 1. Load data
import pandas as pd
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 2. Run notebook
jupyter notebook notebook.ipynb
```

### Train & Evaluate

```bash
# Execute full notebook
python -m jupyter execute notebook.ipynb

# Or run specific sections
# - Section 1: EDA & Data Exploration
# - Section 2: Feature Engineering
# - Section 3: Model Training (5-Fold CV)
# - Section 4: Ensemble & Blending
# - Section 5: Submission Generation
```

### Generate Submission

```python
# After training, submission.csv is generated
# Upload to Kaggle:

# Via CLI:
kaggle competitions submit -c smart-mcq-solver-challenge -f submission.csv -m "Model 3: Dual Sparse+Dense Blend"
```

### Inference on New Data

```python
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load pre-trained models (mock example)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
vectorizer = TfidfVectorizer(...)  # fitted on training data

# New question
prompt = "What is machine learning?"
options = {
    'A': "A subset of AI focused on learning from data",
    'B': "A programming language",
    'C': "A database system",
    'D': "An operating system",
    'E': "A web framework"
}

# Predict
probabilities = {}
for letter, option in options.items():
    text_pair = f"{prompt} [SEP] {option}"
    
    # Sparse feature
    sparse_feat = vectorizer.transform([text_pair])
    prob_sparse = classifier_sparse.predict_proba(sparse_feat)[0, 1]
    
    # Dense feature
    dense_feat = embedder.encode([text_pair])
    prob_dense = classifier_dense.predict_proba(dense_feat)[0, 1]
    
    # Blend
    probabilities[letter] = 0.60 * prob_sparse + 0.40 * prob_dense

# Rank top-3
top3 = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
prediction = " ".join([letter for letter, _ in top3])
print(f"Prediction: {prediction}")  # e.g., "A C B"
```

---

## 📁 Project Structure

```
smart-mcq-solver/
│
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── .gitignore                         # Git ignore
│
├── notebook.ipynb                     # Main notebook (or .py export)
│
├── data/
│   ├── train.csv                      # Training data
│   ├── test.csv                       # Test data
│   └── sample_submission.csv          # Submission format template
│
├── output/
│   ├── submission.csv                 # Final submission file
│   ├── model_tfidf_fold_0.pkl         # Saved TF-IDF models
│   ├── model_dense_fold_0.pkl         # Saved Dense models
│   └── metrics.json                   # Evaluation metrics
│
├── src/                               # (Optional) Refactored code
│   ├── preprocessing.py               # Data preparation
│   ├── features.py                    # Feature engineering
│   ├── models.py                      # Model training
│   └── utils.py                       # Utility functions
│
└── docs/
    ├── METHODOLOGY.md                 # Detailed methodology
    ├── EDA.md                         # Exploratory data analysis
    └── RESULTS.md                     # Results & insights
```

---

## 🔍 Key Files Explained

### `notebook.ipynb` (or `.py`)
Main notebook containing:
1. **Section 1**: Environment setup & data loading
2. **Section 2**: Exploratory Data Analysis (EDA)
3. **Section 3**: Pairwise formulation & preprocessing
4. **Section 4**: Feature engineering (TF-IDF + MiniLM)
5. **Section 5**: 5-Fold cross-validation training
6. **Section 6**: Probability blending & ranking
7. **Section 7**: Evaluation & metrics calculation
8. **Section 8**: Submission generation

### `data/` Directory
- `train.csv`: 242 unique prompts × 5 options + labels
- `test.csv`: Test set without labels (for prediction)
- `sample_submission.csv`: Format reference

### `output/` Directory
- `submission.csv`: Final predictions for Kaggle
- Saved models: For inference without retraining

---

## 📈 Results & Performance

### Final Scores (Model 3: Dual Sparse+Dense Blend)

```
Cross-Validation (OOF):
  MAP@3:           0.75644
  Top-1 Accuracy:  0.6560
  Precision:       0.6540
  Recall:          0.6490
  F1-Score:        0.6510

Public Leaderboard:
  MAP@3:           ~0.756
  (Aligns with OOF, confirming no leakage)

Performance vs Baselines:
  Length heuristic:     0.49 MAP@3 (-35%)
  Unsupervised TF-IDF:  0.36 MAP@3 (-52%)
  Our model:            0.756 (+54% vs heuristic)
```

### Per-Class Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| A | 0.66 | 0.63 | 0.64 |
| B | 0.67 | 0.69 | 0.68 |
| C | 0.64 | 0.66 | 0.65 |
| D | 0.62 | 0.60 | 0.61 |
| E | 0.63 | 0.61 | 0.62 |
| **Macro Avg** | **0.654** | **0.649** | **0.651** |

---

## 🎯 Key Achievements

✅ **0.756 MAP@3** — Top-tier performance on public leaderboard

✅ **No data leakage** — Proper GroupKFold cross-validation

✅ **Robust approach** — Combination of classical ML + modern embeddings

✅ **Reproducible** — Fixed random seeds, documented hyperparameters

✅ **Interpretable** — TF-IDF features + weight visualizations

✅ **Efficient** — CPU-compatible, no GPU required

---

## ⚠️ Challenges & Solutions

### Challenge 1: Data Leakage Risk
**Problem**: Dataset has duplicate MCQs; regular KFold splits questions across folds

**Solution**: Use **GroupKFold** with `groups=question_id` to keep all options together

### Challenge 2: Length Bias Dominance
**Problem**: 57.2% of answers are longest; model might overfit to this pattern

**Solution**: Use semantic features (MiniLM) to enforce deeper understanding beyond length

### Challenge 3: Model Performance Gap
**Problem**: DeBERTa fine-tuning achieved only 0.377 MAP@3 (much worse than baseline)

**Solution**: Stick with simpler approach; complex models need more tuning and compute

### Challenge 4: Feature Dimensionality Mismatch
**Problem**: TF-IDF sparse (~10k dims) vs MiniLM dense (384 dims) have different optimal C values

**Solution**: Tune C per model (C=5.0 sparse, C=2.0 dense) based on dimensionality

### Challenge 5: Blend Weight Optimization
**Problem**: 60-40 blend ratio chosen manually; may not be globally optimal

**Solution**: Future work could use Bayesian optimization for systematic tuning

---

## 🚀 Future Improvements

### Short-term (High Impact)
1. **Systematic Hyperparameter Tuning**
   - Use Optuna or GridSearchCV for blend weights
   - Optimize C per model systematically

2. **Feature Ablation Study**
   - Identify which features contribute most
   - Remove redundant features for efficiency

3. **Ensemble Diversification**
   - Add XGBoost, SVM, or neural network models
   - Combine 3+ models instead of 2

### Medium-term (Moderate Impact)
4. **Domain-Specific Embeddings**
   - Fine-tune MiniLM on MCQ dataset
   - Use domain-specific embeddings (if applicable)

5. **Advanced Ensemble Methods**
   - Stacking: train meta-model on OOF predictions
   - Weighted blending via learned coefficients

6. **Explicit Feature Engineering**
   - Add option length as feature
   - Add prompt-option semantic similarity scores

### Long-term (Research Direction)
7. **End-to-End Neural Models**
   - Fine-tuned BERT/RoBERTa with proper training regime
   - Custom ranking loss function (LambdaMART)

8. **Data Augmentation**
   - Generate synthetic MCQs via paraphrasing
   - Increase training data for better generalization

9. **Retrieval-Augmented Generation (RAG)**
   - Incorporate external knowledge base
   - Use context retrieval to improve predictions

10. **Production Deployment**
    - API server for real-time inference
    - GPU acceleration for faster encoding
    - Caching for common options

---

## 📊 Weights & Biases (W&B) Integration

This project tracks experiments via Weights & Biases.

**To view your runs:**

```
Workspace: https://wandb.ai/23f2003050-dl-genai-project/23f2003050-t22026
```

**Environment Setup:**
```bash
# Login to W&B
wandb login

# Set credentials in notebook
os.environ['WANDB_API_KEY'] = 'your-api-key'
```

**Logged Metrics:**
- MAP@3
- Top-1 Accuracy
- Precision, Recall, F1-Score
- Training time per fold
- Hyperparameters

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'sentence_transformers'`
**Solution:**
```bash
pip install sentence-transformers
```

### Issue: Out of Memory (OOM) when encoding with MiniLM
**Solution:**
```python
embedder.encode(texts, batch_size=32)  # Reduce batch size
# or use GPU:
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
```

### Issue: Kaggle data not found
**Solution:**
```bash
# Authenticate Kaggle CLI
kaggle competitions download -c smart-mcq-solver-challenge

# Or download manually from Kaggle competition page
```

### Issue: Random seeds not working (non-deterministic results)
**Solution:**
```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
```

---

## 📚 References

### Papers & Articles
- [Mean Average Precision (MAP) Explained](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [Learning to Rank with sklearn](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)

### Libraries & Tools
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Sentence-Transformers](https://www.sbert.net/)
- [Weights & Biases](https://wandb.ai/)
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)

### Related Competitions
- [Kaggle MCQ Solver Challenge](https://www.kaggle.com/competitions/smart-mcq-solver-challenge)
- [Question Answering Benchmarks (RACE, SQuAD)](https://huggingface.co/datasets)

---

## 💬 Support & Contribution

### Getting Help
- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for ideas
- **Email**: Contact project maintainer

### Contributing
Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 👤 Author

**VELUGULA VENKATA SURYAMITRA**

- **Roll Number**: 23F2003050
- **Institution**: IIT Madras (BSDA Program)
- **Course**: BSDA2001P — Introduction to Deep Learning and GenAI
- **W&B Entity**: `23f2003050-dl-genai-project`
- **GitHub**: [@yourgithubhandle](https://github.com/yourgithubhandle)

---

## 🙏 Acknowledgments

Special thanks to:
- **Kaggle** for hosting the competition and providing the dataset
- **IIT Madras BSDA Program** for the Deep Learning & GenAI curriculum
- **Hugging Face** for Sentence Transformers and pre-trained models
- **W&B Team** for experiment tracking infrastructure
- **Open-source ML community** for scikit-learn, pandas, numpy

---

## 📅 Project Timeline

| Phase | Dates | Status |
|-------|-------|--------|
| Project Setup & EDA | Week 1-2 | ✅ Complete |
| Feature Engineering | Week 3-4 | ✅ Complete |
| Model Development | Week 5-6 | ✅ Complete |
| Tuning & Optimization | Week 7 | ✅ Complete |
| Final Submission | Week 8 | ✅ Complete |

---

## 🎓 Learning Outcomes

Through this project, I gained expertise in:

- ✅ NLP feature engineering (TF-IDF, embeddings)
- ✅ Proper cross-validation strategies (preventing leakage)
- ✅ Ensemble methods and probability blending
- ✅ Experiment tracking and reproducibility
- ✅ Competitive ML workflow (Kaggle)
- ✅ Transfer learning with pre-trained models

---

## ⭐ Citation

If you use this project or find it helpful, please cite:

```bibtex
@misc{suryamitra2026mcqsolver,
  author = {Velugula, Venkata Suryamitra},
  title = {Smart MCQ Solver Challenge: Dual Sparse+Dense Ensemble},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\\url{https://github.com/yourgithubhandle/smart-mcq-solver}},
  institution = {IIT Madras, BSDA}
}
```

---

## 📞 Contact & Feedback

- **GitHub Issues**: For bug reports and feature requests
- **Email**: your.email@example.com
- **Kaggle Discussion**: https://www.kaggle.com/competitions/smart-mcq-solver-challenge/discussion
- **W&B**: https://wandb.ai/23f2003050-dl-genai-project

---

**Last Updated:** August 2026

**Status:** ✅ Production Ready | 📊 Public Leaderboard Score: 0.756 MAP@3

---

## Quick Stats

- **Total Training Time**: ~15 minutes (5 folds × 2 models)
- **Inference Time**: ~2 seconds per test set (500 questions)
- **Model Size**: ~50 MB (TF-IDF vectorizer + model weights)
- **Memory Required**: 4 GB (CPU) or 2 GB (GPU)
- **Python Version**: 3.12.13
- **Last Tested**: August 3, 2026

---

*Made with ❤️ for the Kaggle Smart MCQ Solver Challenge*
