import pandas as pd
from sklearn.model_selection import train_test_split

from CentralizedSVM import CentralizedSVM
from DistributedSVM import DistributedSVM

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
svm_cen = CentralizedSVM()
svm_cen.train(x_train, y_train)
svm_cen.predict(x_test, y_test)


# Distributed SVM execution
svm_dist = DistributedSVM()
svm_dist.train(x_train, y_train)
svm_dist.predict(x_test, y_test)

# Metrics
svm_cen.metrics()
svm_dist.test_metrics()
svm_dist.trend_metrics()

