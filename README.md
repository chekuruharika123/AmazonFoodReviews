
📦 Amazon Food Reviews Sentiment Prediction
🔹 Project Overview
This project predicts whether an Amazon food review is positive or negative using NLP and machine learning.
It demonstrates data preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation.

🔹 Problem Statement
Online reviews influence buying decisions.
Can we automatically classify review sentiment to identify genuine feedback?
Input: Review text | Output: Predicted sentiment (Positive / Negative)
🔹 Dataset
Source: [Kaggle / Amazon Food Reviews dataset]
Size: ~50,000 reviews
Features: Review text, Rating, Helpful votes
🔹 Approach
1. Data Preprocessing
Cleaned text: removed special characters, stopwords, lowercasing
Tokenization & lemmatization
2. Feature Engineering
TF-IDF vectorization
Bag-of-Words representation
3. Model Training & Hyperparameter Tuning
Logistic Regression → tuned C and solver → accuracy 85%
Hyperparameter tuning improved model performance and shows practical ML skills.

🔹 Evaluation
Metrics: Accuracy & F1-score
Confusion matrix
Insight: Review length and sentiment keywords strongly predict positive ratings
🔹 Visual Insights
word clouds
histograms
line plots

🔹 Tools & Libraries
Python: Pandas, NumPy
NLP: NLTK, Scikit-learn
ML: Scikit-learn (Logistic Regression, Random Forest)
Visualization: Matplotlib, Seaborn, WordCloud
🔹 Future Work
Deploy as a web app for real-time prediction
Experiment with deep learning (LSTM, Transformers)
Expand to multilingual reviews
🔹 Key Takeaways
Hyperparameter tuning is crucial for performance
Preprocessing and feature engineering significantly affect results
Project demonstrates ability to build end-to-end NLP pipelines
