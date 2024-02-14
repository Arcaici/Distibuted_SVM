import matplotlib.pyplot as plt
import pandas as pd
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

pd.set_option('display.max_columns', None)
# Read dataset
df = pd.read_csv('Datasets/apple_quality.csv')
df = df.drop(['A_id'], axis=1)

# Removing Nan Values
df = df.dropna()
df['Quality'] = df['Quality'].apply(lambda x: -1 if x == 'bad' else 1)

# Split the dataset into training and test sets
x = df.iloc[:, :6].values
y = df['Quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Centralized SVM execution
svm_cen = CentralizedSVM(lambda_val=0.01, verbose=False)
svm_cen.fit(x_train, y_train)
svm_cen.predict(x_test, y_test)


# Distributed SVM execution
svm_dist = DistributedSVM(rho=0.1,lambda_val=0.01, verbose=False, early_stopping=True)
svm_dist.fit(x_train, y_train)
svm_dist.predict(x_test, y_test)

# Metrics
svm_cen.metrics()
svm_dist.test_metrics()
svm_dist.trend_metrics()

# comparing plots
comparing_metrics(svm_cen, svm_dist)

