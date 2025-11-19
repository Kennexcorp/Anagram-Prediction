# Anagram Prediction

This project explores different methods for detecting anagrams, ranging from a simple algorithmic approach to a machine learning-based solution.

## Overview

The project compares two main approaches:
1.  **Algorithmic Solution**: A Python function using `collections.Counter` to check if two strings have the same character counts.
2.  **Machine Learning Solution**: A Logistic Regression model trained on a dataset of anagram and non-anagram pairs.

## Learning Outcomes

By working through this project, you will gain insights into:

-   **Algorithmic vs. Machine Learning Approaches**: Understanding when to use a deterministic rule-based solution (which is O(n) and 100% accurate for this problem) versus a probabilistic ML model.
-   **Synthetic Data Generation**: Learning how to generate a balanced dataset of positive (anagrams) and negative (random pairs) samples to train a binary classifier.
-   **Feature Engineering for NLP**: utilizing `CountVectorizer` with a character-level analyzer (`analyzer='char'`) to convert strings into numerical vectors that represent character frequencies—the core feature of an anagram.
-   **Model Training & Evaluation**: Practical experience in splitting data into training and testing sets, training a Logistic Regression model, and evaluating its performance using accuracy scores and cross-validation.
-   **Understanding Model Behavior**: Observing how an ML model "learns" the concept of anagrams by weighting character counts, and identifying potential pitfalls where the model might rely on simple heuristics or biases in the training data.

## Files

-   `anagram_checker.ipynb`: The main Jupyter Notebook containing the code for data generation, model training, and evaluation.
-   `anagram_dictionary.csv` / `anagram_dictionary.txt`: A dataset of known anagram pairs used for training.
-   `wordlist.csv` / `wordlist.txt`: A list of words used to generate non-anagram pairs for the negative class.
-   `anagram_model.pkl`: The trained Logistic Regression model.
-   `pipe_anagram_model.pkl`: A pipeline model (likely including vectorization and the classifier).

## Usage

To run the project:

1.  Ensure you have Jupyter Notebook or JupyterLab installed.
2.  Install the required dependencies (e.g., `pandas`, `scikit-learn`, `numpy`).
3.  Open `anagram_checker.ipynb`.
4.  Run the cells sequentially to see the data generation, model training, and results.

## Methodology

The machine learning approach involves:
1.  **Data Generation**: Creating a balanced dataset of positive samples (anagrams) and negative samples (random word pairs).
2.  **Feature Extraction**: Using `CountVectorizer` to convert word pairs into numerical vectors based on character counts.
3.  **Model Training**: Training a Logistic Regression model to classify pairs as anagrams or not.
4.  **Evaluation**: Assessing the model's accuracy on a test set.
