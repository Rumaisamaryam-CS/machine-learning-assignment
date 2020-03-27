#MACHINE LEARNING ASSIGNMENT2
#RUMAISA MARYAM
# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('headweightset.csv')
A = dataset.iloc[:, 2:-1].values
B = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size = 0.2, random_state = 0)


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(A, B)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
A_poly = poly_reg.fit_transform(A)
poly_reg.fit(A_poly, B)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(A_poly, B)

# Visualising the Linear Regression results
plt.scatter(A, B, color = 'red')
plt.plot(A, lin_regressor.predict(A), color = 'Black')
plt.title('Brain Weight VS Head Size (Linear Regression)')
plt.xlabel('HEAD SIZE(in (cm^3))')
plt.ylabel('BRAIN WEIGHT')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(A, B, color = 'red')
plt.plot(A, lin_reg_2.predict(poly_reg.fit_transform(A)), color = 'Black')
plt.title('Brain Weight VS Head Size(Polynomial Regression)')
plt.xlabel('HEAD SIZE(in (cm^3))')
plt.ylabel('BRAIN WEIGHT')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
A_grid = np.arange(min(A), max(A), 0.1)
A_grid = A_grid.reshape((len(A_grid), 1))
plt.scatter(A, B, color = 'red')
plt.plot(A_grid, lin_reg_2.predict(poly_reg.fit_transform(A_grid)), color = 'Black')
plt.title('Brain Weight VS Head Size (Polynomial Regression)')
plt.xlabel('HEAD SIZE(in (cm^3))')
plt.ylabel('BRAIN WEIGHT')
plt.show()

# Predicting a new result with Linear Regression
lin_regressor.predict([[6]])

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6]]))