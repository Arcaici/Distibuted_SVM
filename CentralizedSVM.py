import numpy as np
import math
import seaborn as sns
import pandas as pd
import cvxpy as cp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

class CentralizedSVM():

    def __init__(self, lambda_val = 1e-2, verbose = True, real= True):

        # param
        self.lambda_val = lambda_val
        self.verbose = verbose
        self.real = real

        # estimated param
        self.w_c = 0
        self.b_c = 0

        # metrics
        self.accuracy = 0
        self.fpr = 0
        self.tpr = 0
        self.roc_auc = 0
        self.cm = 0
    def fit(self, x_train, y_train):

        # Params
        m = x_train.shape[0]
        n = x_train.shape[1]

        if self.real:
            y_train_np = y_train.values
            y_train_reshaped = y_train_np.reshape(1, -1)
        else:
            y_train_reshaped = y_train.T

        # Cost function and variables
        A = np.hstack((x_train * y_train_reshaped.T, y_train_reshaped.T))
        C = np.identity(n + 1)
        C[n, n] = 0
        x_v = cp.Variable((n + 1, 1))
        loss = cp.sum(cp.pos(1 - A @ x_v))
        reg = cp.norm(C @ x_v, 1)
        prob = cp.Problem(cp.Minimize(loss / m + self.lambda_val * reg))

        # Solver
        prob.solve(solver=cp.ECOS, verbose= self.verbose)
        self.w_c = x_v.value[:n]
        self.b_c = x_v.value[-1]

        print("#________ DONE TRAIN Centralized________#")

    def predict(self, x_test, y_test):

        # Prediction
        self.w_c = self.w_c.reshape(-1, 1)
        y_pred = np.sign(np.dot(x_test, self.w_c) + self.b_c)

        # Save metrics
        self.accuracy = accuracy_score(y_test, y_pred)
        self.fpr, self.tpr, _ = roc_curve(y_test, y_pred)
        self.roc_auc = auc(self.fpr, self.tpr)
        self.cm = confusion_matrix(y_test, y_pred)


    def metrics(self):

        #Accuracy
        print("Accuracy (Centralized):", self.accuracy)
        # ROC and AUC
        print("AUC (Centralized):", self.roc_auc)
        plt.figure()
        plt.plot(self.fpr, self.tpr, color='blue', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve SVM Centralized')
        plt.show()

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.cm, annot=True, cmap='Blues', fmt='g', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Centralized)')
        plt.show()

