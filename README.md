# Project Introduction

This project aims to implement and compare two versions of the Support Vector Machine (SVM) model: a standard SVM model and a distributed SVM model using the Alternative Directions Multipliers Method (ADMM). The comparison focuses on examining their convergence and assessing whether the distributed version can match the performance of the centralized one using real data. The project also involves tuning model parameters, with a focus on understanding how ADMM parameters influence the convergence of the distributed model.

## Parameters and Score Metrics

The project's parameters include:
- **lambda**: Regularization parameter used for fine-tuning the SVM model in the centralized version and reused in the distributed version.
- **rho**: Parameter indicating convergence within the distributed version, tested to determine the fastest convergence.
- **Stopping Criteria (or Early Stopping)**: Option to stop the model when it reaches convergence instead of waiting for a predetermined iteration value.

General metrics used for comparing model performance include accuracy, confusion matrix, ROC and AUC scores, loss trend, and disagreement trend.

## Dataset

Two datasets are used: a synthetic dataset for verifying code implementation and a real dataset called "Apple Quality" obtained from Kaggle. The purpose of the project is not to implement a model for a specific task but to develop a framework for distributed optimization and compare implementations in terms of potential company infrastructure.

# Support Vector Machine

The Support Vector Machine (SVM) is a classification model that constructs a hyperplane decision boundary to separate data into different regions. The SVM aims to maximize the margin between the hyperplane and the support vectors, which are data points closest to the decision boundary. The mathematical formulation of SVM is presented, along with the implementation of the distributed version using ADMM.

## Alternative Directions Multipliers Method

ADMM is a mathematical framework used to implement distributed versions of models. It involves dividing the loss function and regularization into separate steps to perform optimization procedures while maintaining consistency. The distributed SVM model using ADMM is explained, highlighting the convergence process and parameter tuning.

# SVM Code Implementation

The project is implemented in Python using CVXPY, a modeling language for convex optimization problems. The Centralized SVM and Distributed SVM (by Samples) are presented, along with the implementation details of the ADMM framework.

## Centralized SVM

A simple SVM implementation is provided, utilizing CVXPY for convex optimization. The training phase of the Centralized SVM class is explained, including the management of label vectors and the definition of cost function variables.

## Distributed SVM (by Samples)

The implementation of the ADMM framework for distributed SVM is described, showcasing the steps involved in the optimization process. Details about the stopping criteria and convergence monitoring are provided.

# Model Selection Results

The model selection process involves grid search for parameter tuning and examining convergence behavior. The results of the grid search for both centralized and distributed models are discussed, along with the impact of ADMM parameters on convergence.

## ADMM Model Selection

The influence of different rho values on loss convergence and model performance is analyzed. The process of selecting the optimal rho value is explained, considering convergence speed and model scores.

## ADMM Stopping Criteria

The implementation of stopping criteria in the ADMM framework is tested, and its effect on loss convergence and iteration count is examined. The process of choosing the optimal rho value with stopping criteria is discussed.

# Comparison and Conclusions

The performance of the centralized and distributed models is compared using various metrics, including accuracy, confusion matrix, ROC and AUC scores. The convergence behavior and parameter tuning results are summarized, and future research directions are suggested.
