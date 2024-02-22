# Distibuted_SVM

This project aims to implement and compare two versions of the Support Vector Machine (SVM) model: a standard SVM model and a distributed SVM model using the Alternative Directions Multipliers Method (ADMM). The comparison focuses on examining their convergence and assessing whether the distributed version can match the performance of the centralized one using real data. The project also involves tuning model parameters, with a focus on understanding how ADMM parameters influence the convergence of the distributed model.

## Parameters and Score Metrics

The project's parameters include:
- **lambda**: Regularization parameter used for fine-tuning the SVM model in the centralized version and reused in the distributed version.
- **rho**: Parameter indicating convergence within the distributed version, tested to determine the fastest convergence.
- **Stopping Criteria (or Early Stopping)**: Option to stop the model when it reaches convergence instead of waiting for a predetermined iteration value.

General metrics used for comparing model performance include accuracy, confusion matrix, ROC and AUC scores, loss trend, and disagreement trend.

## Dataset

Two datasets are used: a synthetic dataset for verifying code implementation and a real dataset called [Apple Quality](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality) obtained from Kaggle. The purpose of the project is not to implement a model for a specific task but to develop a framework for distributed optimization and compare implementations in terms of potential company infrastructure.

# Alternative Directions Multipliers Method

ADMM is a mathematical framework used to implement distributed versions of models. It involves dividing the loss function and regularization into separate steps to perform optimization procedures while maintaining consistency. The distributed SVM model using ADMM is explained, highlighting the convergence process and parameter tuning.

## SVM Code Implementation

The project is implemented in Python using CVXPY, a modeling language for convex optimization problems. The Centralized SVM and Distributed SVM (by Samples) are presented, along with the implementation details of the ADMM framework.

# Model Selection Results

The model selection process involves grid search for parameter tuning and examining convergence behavior. The results of the grid search for both centralized and distributed models are discussed, along with the impact of ADMM parameters on convergence.

<div>
    <img src="https://github.com/Arcaici/Distibuted_SVM/blob/master/images/ADMM_SVM/losses_convergence.png" alt="models comparison over rho values" width="400" />
    <img src="https://github.com/Arcaici/Distibuted_SVM/blob/master/images/Stopping_criteria/early_stopping.png" alt="models comparison over rho values with stopping criteria" width="400" />
</div>

In these examples, we can observe the distributed version exhibiting different convergence behaviors using various step-size values and how applying a stopping criterion can influence their behavior.

# Comparison and Conclusions

The performance of the centralized and distributed models is compared using various metrics, including accuracy, confusion matrix, ROC, and AUC scores. The project demonstrates how both finely-tuned models converge to the same performance level.
<div>
    <img src="https://github.com/Arcaici/Distibuted_SVM/blob/master/images/Final%20Results/ROC_AUC_comparison.png" alt="Distributed and Centralized ROC models comparison" width="400" />
    <img src="https://github.com/Arcaici/Distibuted_SVM/blob/master/images/Final%20Results/Accuracy_comparison.png" alt="Centralized and Distributed ROC models comparison" width="400" />
</div>

For further information check the complete guide [here]()
