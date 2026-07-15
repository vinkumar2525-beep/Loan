# Loan Approval Prediction System

A machine learning classification system to predict loan approval with multiple algorithms and an interactive Streamlit web application.

## 🎯 Models Trained

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Gradient Boosting** ⭐ | **91.85%** | 93.06% | 93.60% | 93.33% |
| Random Forest | 91.00% | 92.40% | 92.86% | 92.63% |
| Naive Bayes | 87.85% | 87.47% | 93.43% | 90.35% |
| Logistic Regression | 78.30% | 78.24% | 89.16% | 83.35% |
| Support Vector Machine | 77.50% | 75.84% | 92.53% | 83.36% |
| K-Nearest Neighbors | 74.25% | 75.38% | 85.71% | 80.22% |

**Best Model:** Gradient Boosting with **91.85% accuracy**

## 📊 Dataset
- **File:** `loan.csv`
- **Records:** 10,000 loan applications
- **Features:** 17 input features + target variable
- **Target:** Loan Approved (0 = Rejected, 1 = Approved)

## 🚀 Features

1. **Model Training** (`model_training.py`)
   - 6 different classification algorithms
   - Comprehensive evaluation metrics
   - Pickle files saved for deployment

2. **Interactive Web App** (`loan_app.py`)
   - Streamlit-based interface
   - Real-time predictions
   - Model comparison dashboard
   - Probability scores

## 📋 Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- streamlit
- pyarrow

## 🔧 Installation & Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\Activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Models
```bash
python model_training.py
```

### 4. Run Streamlit App
```bash
streamlit run loan_app.py
```

The app will be available at: **http://localhost:8501**

## 📝 Usage

1. Open the Streamlit app
2. Fill in the loan application details:
   - Personal information (Age, Gender, Marital Status, etc.)
   - Financial details (Income, Loan Amount, etc.)
   - Loan specifics (Term, Purpose, Credit Score, etc.)
3. Click "🔮 Predict Loan Approval"
4. View predictions from all models with approval probabilities

## 📁 Project Structure

```
Loan/
├── loan.csv                    # Dataset
├── model_training.py           # Model training script
├── loan_app.py                 # Streamlit web application
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
├── best_model.pkl              # Best model (Gradient Boosting)
├── all_models.pkl              # All trained models
├── label_encoders.pkl          # Feature encoders
├── feature_names.pkl           # Feature names
├── results.pkl                 # Training results
├── model_results.csv           # Model performance report
└── README.md                   # This file
```

## 🔬 Model Details

### Gradient Boosting (Best Model)
- Accuracy: 91.85%
- Precision: 93.06%
- Recall: 93.60%
- Great balance between precision and recall
- Handles imbalanced data well

### Other Models Trained
- Random Forest: 91.00% accuracy
- Naive Bayes: 87.85% accuracy
- Logistic Regression: 78.30% accuracy
- Support Vector Machine: 77.50% accuracy
- K-Nearest Neighbors: 74.25% accuracy

## 🔐 GitHub Push Instructions

To complete pushing to GitHub, follow these steps:

### Option 1: Using Personal Access Token (Recommended)
1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Generate a new token with `repo` scope
3. Run: `git push -u origin main`
4. When prompted for password, use your personal access token

### Option 2: Using SSH
1. Generate SSH keys: `ssh-keygen -t ed25519 -C "your_email@example.com"`
2. Add SSH key to GitHub account
3. Change remote: `git remote set-url origin git@github.com:vinkumar2525-beep/Loan.git`
4. Push: `git push -u origin main`

### Option 3: Using GitHub CLI
```bash
gh auth login
git push -u origin main
```

## 📊 Model Performance Breakdown

### Dataset Split
- Training: 8,000 samples (80%)
- Testing: 2,000 samples (20%)
- Target Distribution: 60.9% Approved, 39.1% Rejected

### Evaluation Metrics
- **Accuracy:** Overall correctness
- **Precision:** Accuracy of positive predictions
- **Recall:** Coverage of actual positives
- **F1-Score:** Harmonic mean of precision and recall

## 🎓 Learning Outcomes

This project demonstrates:
- Data preprocessing and feature encoding
- Multiple classification algorithms
- Model evaluation and comparison
- Web application deployment with Streamlit
- Git and GitHub workflow
- End-to-end ML pipeline

## 📞 Contact & Support

For questions or issues, please check the GitHub repository.

---

**Status:** ✅ Ready for Deployment  
**Last Updated:** 2026-07-15  
**Maintained by:** Loan Classification Project
