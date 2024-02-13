import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import product
from sklearn.metrics import accuracy_score
from CentralizedSVM import CentralizedSVM
from DistributedSVM import DistributedSVM


def centralized_grid_search(lambda_values, x_train, y_train, x_val, y_val):
    best_lambda = None
    best_score = -1

    for lambda_val in lambda_values:

        model = CentralizedSVM(lambda_val=lambda_val, verbose=False)
        model.fit(x_train, y_train)
        model.predict(x_val, y_val)
        score = model.accuracy

        if score > best_score:
            best_score = score
            best_lambda = lambda_val

    return best_lambda, best_score

def distributed_grid_search(rho_values, lambda_val, x_train, y_train, x_val, y_val, early_stopping = False):
    best_rho = None
    best_score = -1
    max_iter = {}
    scores = {}
    loss_convergence = {}

    for rho in rho_values:
        if early_stopping:
            model = DistributedSVM(rho=rho,lambda_val=lambda_val, verbose=False, early_stopping=True)
        else:
            model = DistributedSVM(rho=rho,lambda_val=lambda_val, verbose=False)
        model.fit(x_train, y_train)
        model.predict(x_val, y_val)
        score = model.accuracy
        max_iter[f'{rho}'] = model.iter
        scores[f'{rho}'] = score
        loss_convergence[f'{rho}'] = model.LOSS[:,0]

        if score > best_score:
            best_score = score
            best_rho = rho

        print(f'done validation of rho: {rho}')

    return best_rho, best_score, loss_convergence, scores, max_iter

pd.set_option('display.max_columns', None)

# Read dataset
df = pd.read_csv('Datasets/apple_quality.csv')
df = df.drop(['A_id'], axis=1)

# Removing Nan Values
df = df.dropna()
df['Quality'] = df['Quality'].apply(lambda x: -1 if x == 'bad' else 1)

# Split the dataset into training, validation and test sets
x = df.iloc[:, :6].values
y = df['Quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train_g, x_val, y_train_g, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Perform grid search on the dataset for different lambda values in Centralized version
lambda_values = [0.0001, 0.001, 0.01, 0.1, 1]
best_lambda, best_score_c = centralized_grid_search(lambda_values, x_train_g,y_train_g, x_val, y_val)

# results and tuned model
print("best_lambda: " ,best_lambda)
print("best_score: ", best_score_c)

# Best Centralized SVM model
svm_cen = CentralizedSVM(lambda_val=best_lambda, verbose=False)
svm_cen.fit(x_train, y_train)
svm_cen.predict(x_test, y_test)
svm_cen.metrics()

# Perform grid search on the dataset for different rho values in Distributed version
rho_values = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
best_rho, best_score_d, loss_convergence, scores, iter = distributed_grid_search(rho_values, best_lambda, x_train_g, y_train_g, x_val, y_val)

print("best_rho: " ,best_rho)
print("best_score: ", best_score_d)

# Plot Loss convergence and scores
for (rho_1, loss), (rho_2, score) in zip(loss_convergence.items(), scores.items()):
    plt.plot(loss, label=f'rho: {rho_1} score: {score}')

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss convergence for each model with different rho')
plt.legend()
plt.show()

# Perform grid search with early_stopping Distributed version
best_rho, best_score_d, loss_convergence, scores, iter = distributed_grid_search(rho_values, best_lambda, x_train_g, y_train_g, x_val, y_val, early_stopping=True)

print("best_rho: " ,best_rho)
print("best_score: ", best_score_d)

# Plot Loss convergence and scores
for (rho_1, loss), (rho_2, score) in zip(loss_convergence.items(), scores.items()):
    plt.plot(loss, label=f'rho: {rho_1} score: {score}')

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss convergence with different rho using stopping criteria')
plt.legend()
plt.show()

# iteration for convergence based on stopping criteria
for (rho_1, it), (rho_2, score) in zip(iter.items(), scores.items()):
    plt.bar(rho_1, it, label=f'rho: {rho_2} score: {score}')

for i, v in enumerate(iter.values()):
    plt.text(i, v + 0.1, f"{v:.2f}", ha='center', va='bottom')

# Add legend using zip to combine labels and values
plt.legend()
plt.xlabel('rho')
plt.ylabel('iter convergence')
plt.title('Models iteration to convergence')
plt.show()

# Best Distributed SVM model
svm_dist = DistributedSVM(rho=best_rho,lambda_val=best_lambda, verbose=False, early_stopping= True)
svm_dist.fit(x_train, y_train)
svm_dist.predict(x_test, y_test)
svm_dist.test_metrics()
svm_dist.trend_metrics()