import numpy as np
import math
import seaborn as sns
import pandas as pd
import cvxpy as cp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

class DistributedSVM():

    def __init__(self,N=20, n_iter=500, rho= 1, lambda_val = 1e-2, verbose = True):

        # param
        self.N = N
        self.n_iter = n_iter
        self.rho = rho
        self.lambda_val = lambda_val
        self.verbose = verbose

        # estimated param
        self.w_c = 0
        self.b_c = 0
        self.LOSS = 0
        self.D = 0

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
        y_train_np = y_train.values
        y_train_reshaped = y_train_np.reshape(1, -1)

        # Cost function Variables
        A = np.hstack((x_train * y_train_reshaped.T, y_train_reshaped.T))
        n_samples = math.floor(A.shape[0] / self.N)
        X = np.zeros((self.n_iter, self.N, n + 1))
        Z = np.zeros((self.n_iter, n + 1))
        U = np.zeros((self.n_iter, self.N, n + 1))
        self.LOSS = np.zeros((self.n_iter, self.N))
        self.D = 0
        for k in range(0, self.n_iter - 1, 1):

            # Step 1
            count = 0
            for i in range(self.N):
                x_cp = cp.Variable(n + 1)
                loss = cp.sum(cp.pos(np.ones(n_samples) - A[count:count + n_samples, :] @ x_cp))
                reg = cp.sum_squares(x_cp - Z[k, :] + U[k, i, :])
                aug_lagr = loss / m + (self.rho / 2) * reg
                prob = cp.Problem(cp.Minimize(aug_lagr))
                prob.solve(solver=cp.ECOS, verbose=self.verbose)
                X[k + 1, i, :] = x_cp.value
                for j in range(n_samples):
                    cost = 1 - np.inner(A[count + j, :], X[k + 1, i, :])
                    if cost > 0:
                        self.LOSS[k + 1, i] += cost
                self.LOSS[k + 1, i] += self.rho / 2 * np.linalg.norm(X[k + 1, i, :] - Z[k, :] + U[k, i, :]) ** 2

                count += n_samples

            # Step 2
            mean_X = np.zeros(n + 1)
            mean_U = np.zeros(n + 1)
            for i in range(self.N):
                mean_X += X[k + 1, i, :]
                mean_U += U[k, i, :]
            mean_X = 1 / self.N * mean_X
            mean_U = 1 / self.N * mean_U

            for i in range(n + 1 - 1):
                if mean_X[i] + mean_U[i] > self.lambda_val / (self.N * self.rho):
                    Z[k + 1, i] = mean_X[i] + mean_U[i] - self.lambda_val / (self.N * self.rho)
                elif mean_X[i] + mean_U[i] < - self.lambda_val / (self.N * self.rho):
                    Z[k + 1, i] = mean_X[i] + mean_U[i] + self.lambda_val / (self.N * self.rho)
                else:
                    Z[k + 1, i] = 0
            Z[k + 1, n] = mean_X[n] + mean_U[n]

            # Step 3
            for i in range(self.N):
                U[k + 1, i, :] = U[k, i, :] + X[k + 1, i, :] - Z[k + 1, :]

            # Disagrement
            P = np.empty((X.shape[2], 0))
            for i in range(self.N):
                p = X[k+1, i, :] - mean_X
                P = np.concatenate((P, p[:, np.newaxis]), axis=1)
            dk = np.sum(np.square(P))
            self.D = np.append(self.D, dk)

        print("#________ DONE TRAIN Distributed________#")
        self.w_c = X[self.n_iter-1,1,:n]
        self.b_c = X[self.n_iter-1,1,-1]

    def predict(self, x_test, y_test):

        # Prediction
        self.w_c = self.w_c.reshape(-1, 1)
        y_pred = np.sign(np.dot(x_test, self.w_c) + self.b_c)
        self.accuracy = accuracy_score(y_test, y_pred)

        # Save metrics
        self.fpr, self.tpr, _ = roc_curve(y_test, y_pred)
        self.roc_auc = auc(self.fpr, self.tpr)
        self.cm = confusion_matrix(y_test, y_pred)


    def test_metrics(self):
        
        # Accuracy of last iteration
        print("Accuracy (Distributed):", self.accuracy)
        
        # ROC and AUC of last iteration
        print("AUC (Distributed):", self.roc_auc)
        plt.figure()
        plt.plot(self.fpr, self.tpr, color='blue', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve SVM Distributed')
        plt.show()

        # Confution Matrix of last iteration 
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.cm, annot=True, cmap='Blues', fmt='g', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Distributed)')
        plt.show()


    def trend_metrics(self):
        # Loss trend over one agent distribution
        plt.plot(np.linspace(0, self.n_iter, self.n_iter), self.LOSS[:, 0])
        plt.ylabel("LOSS", fontsize=16)
        plt.xlabel("n° iterations", fontsize=16)
        plt.title("Loss trend")
        plt.show()

        # Disagreement between agents
        plt.semilogy(self.D)
        plt.xlabel('Iterations')
        plt.ylabel('Disagreement')
        plt.title(f"Iterations = {self.n_iter} | Processor = {self.N} | rho = {self.rho}")
        plt.show()
