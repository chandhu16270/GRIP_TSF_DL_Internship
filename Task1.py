# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("StudyHoursVsScores.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Fitting Simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)

# Predicting the Test set results
y_pred = regressor.predict([[9.25]])

# Visualizing the training set results
plt.scatter(X, y, color = 'red')
plt.scatter([[9.25]],regressor.predict([[9.25]]), color = 'b')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Experience Vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
