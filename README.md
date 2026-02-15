Credit Card Fraud Detection – Machine Learning Assignment 2
1. Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models to detect fraudulent credit card transactions. Credit card fraud detection is a critical real-world problem due to the highly imbalanced nature of transaction data and the high cost associated with false negatives (missed frauds).

This project implements six classification models, evaluates them using multiple performance metrics, and deploys the models using a Streamlit web application for interactive prediction and evaluation.

2. Dataset Description
* Dataset Name: Credit Card Fraud Detection
* Source: Kaggle (European cardholders dataset)
* Total Transactions: 284,807
* Fraudulent Transactions: 492
* Non-Fraud Transactions: 284,315
* Features:
    V1 to V28: PCA-transformed numerical features
* Amount: Transaction amount
* Target Variable:
    Class
      0 → Non-fraud
      1 → Fraud

Due to extreme class imbalance in the original dataset, a balanced dataset was created using random under-sampling for fair model comparison and faster deployment. The final balanced dataset contains 984 samples with equal fraud and non-fraud instances.

3. Machine Learning Models Used

The following six classification models were implemented on the same dataset:

* Logistic Regression
* Decision Tree Classifier
* K-Nearest Neighbors (KNN)
* Naive Bayes (Gaussian)
* Random Forest (Ensemble Model)
* XGBoost (Ensemble Model)

4. Model Evaluation Metrics

Each model was evaluated using the following metrics:
* Accuracy
* ROC-AUC Score
* Precision
* Recall
* F1-Score
* Matthews Correlation Coefficient (MCC)

| Model               | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | -------- | ------ |
| Logistic Regression | 0.9340   | 0.9840 | 0.9474    | 0.9184 | 0.9326   | 0.8684 |
| Decision Tree       | 0.9188   | 0.9188 | 0.9100    | 0.9286 | 0.9192   | 0.8377 |
| KNN                 | 0.8985   | 0.9517 | 0.9432    | 0.8469 | 0.8925   | 0.8010 |
| Naive Bayes         | 0.9086   | 0.9488 | 0.9545    | 0.8571 | 0.9032   | 0.8214 |
| Random Forest       | 0.9492   | 0.9793 | 0.9783    | 0.9184 | 0.9474   | 0.9001 |
| XGBoost             | 0.9442   | 0.9812 | 0.9677    | 0.9184 | 0.9424   | 0.8894 |

5. Model Performance Observations
 
| Model               | Observation                                                                                                |
| ------------------- | ---------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Achieved high AUC, indicating strong linear separability of PCA-transformed features.                      |
| Decision Tree       | Performed reasonably well but showed slight overfitting compared to ensemble models.                       |
| KNN                 | High precision but lower recall, indicating conservative fraud prediction behavior.                        |
| Naive Bayes         | Moderate performance; independence assumption limits effectiveness.                                        |
| Random Forest       | Best overall performer with highest F1-score and MCC, showing strong balance between precision and recall. |
| XGBoost             | Achieved the highest AUC and competitive performance, close to Random Forest.                              |

Conclusion : Random Forest and XGBoost demonstrated superior performance for fraud detection, with strong balance between precision and recall.
