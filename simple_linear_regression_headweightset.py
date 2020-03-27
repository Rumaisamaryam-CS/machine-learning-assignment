#MACHINE LEARNING ASSIGNMENT2
#RUMAISA MARYAM
# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the headweightset dataset
dataset = pd.read_csv('headweightset.csv')
A = dataset.iloc[:,2:3].values
B = dataset.iloc[:, 3:4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressing = LinearRegression()
regressing.fit(A_train, B_train)

# Predicting the Test set results
B_pred = regressing.predict(A_test)

# Visualising the Training set results
plt.scatter(A_train, B_train, color = 'red')
plt.plot(A_train, regressing.predict(A_train), color = 'black')
plt.title('Brain Weight VS Head Size (Training set)')
plt.xlabel('HEAD SIZE(in (cm^3))')
plt.ylabel('BRAIN WEIGHT')
plt.show()

# Visualising the Test set results
plt.scatter(A_test, B_test, color = 'red')
plt.plot(A_train, regressing.predict(A_train), color = 'black')
plt.title('Brain Weight VS Head Size (Test set)')
plt.xlabel('HEAD SIZE(in (cm^3))')
plt.ylabel('BRAIN WEIGHT')
plt.show()