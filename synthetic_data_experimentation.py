import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from CentralizedSVM import CentralizedSVM
from DistributedSVM import DistributedSVM

def comparing_metrics(svm_cen, svm_dist):
    # Accuracy
    plt.bar("SVM",svm_cen.accuracy, label=f'SVM lambda: {svm_cen.lambda_val}')
    plt.bar("ADMM SVM",svm_dist.accuracy, label=f'ADMM SVM rho: {svm_dist.rho}')
    plt.legend()
    plt.text(0, svm_cen.accuracy + 0.1, f"{svm_cen.accuracy:.3f}", ha='center', va='bottom')
    plt.text(1, svm_dist.accuracy + 0.1, f"{svm_dist.accuracy:.3f}", ha='center', va='bottom')
    plt.ylabel('Accuracy')
    plt.title('Centralized and Distibuted SVM accuracy compared')
    plt.show()

    # ROC & AUC
    plt.figure()
    plt.plot(svm_cen.fpr, svm_cen.tpr, color='blue', lw=2, label=f'SVM AUC: {svm_cen.roc_auc}')
    plt.plot(svm_dist.fpr, svm_dist.tpr, color='red', lw=2, label=f'SVM AUC: {svm_dist.roc_auc}')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve SVM and ADMM SVM')
    plt.show()

# Generate synthetic data
n = 30
m = 9000
m_t = 900

beta_true = np.random.randn(n,1)
offset = np.random.randn(1)
x_train = np.random.normal(0, 5, size=(m,n))
y_train = np.sign(x_train.dot(beta_true) + offset)
x_test = np.random.normal(0, 5, size=(m_t ,n))
y_test = np.sign(x_test.dot(beta_true) + offset)

additional_features_train_1 = np.random.normal(0, 5, size=(m, n))
additional_features_train_2 = np.random.normal(2, 7, size=(m, n))
additional_features_train_3 = np.random.normal(1, 9, size=(m, n))
additional_features_test_1 = np.random.normal(0, 5, size=(m_t, n))
additional_features_test_2 = np.random.normal(2, 7, size=(m_t, n))
additional_features_test_3 = np.random.normal(1, 9, size=(m_t, n))

x_train = np.concatenate((x_train, additional_features_train_1), axis=1)
x_train = np.concatenate((x_train, additional_features_train_2), axis=1)
x_train = np.concatenate((x_train, additional_features_train_3), axis=1)
x_test = np.concatenate((x_test, additional_features_test_1), axis=1)
x_test = np.concatenate((x_test, additional_features_test_2), axis=1)
x_test = np.concatenate((x_test, additional_features_test_3), axis=1)

beta = np.append(beta_true, offset).reshape(31) # param to estimate


# Centralized SVM execution
svm_cen = CentralizedSVM(lambda_val=0.01, verbose=False, real = False)
svm_cen.fit(x_train, y_train)
svm_cen.predict(x_test, y_test)


# Distributed SVM execution
svm_dist = DistributedSVM(rho=0.1,lambda_val=0.01, verbose=False, early_stopping=True, real = False)
svm_dist.fit(x_train, y_train)
svm_dist.predict(x_test, y_test)

# Metrics
svm_cen.metrics()
svm_dist.test_metrics()
svm_dist.trend_metrics()

# comparing plots
comparing_metrics(svm_cen, svm_dist)