import numpy as np
import matplotlib.pyplot as plt

# Load the data (100 rows, 3 columns: x, y, label)
data = np.loadtxt("ex2data1.txt", delimiter=",")

# Take first 100 rows
data = data[:100, :]

# Extract x and y features (columns 0 and 1)
X = data[:, 0:2]

# Optionally, extract label if needed
# y_label = data[:, 2]

# For simplicity, let's just use x1 as X and x2 as Y (or vice versa depending on your goal)
# If you meant to do linear regression on x1 -> x2:
X_feature = X[:, 0]  # shape (100,)
Y_target = X[:, 1]   # shape (100,)

# Reshape X_feature to a column vector for matrix ops
X_feature = X_feature.reshape(-1, 1)
Y_target = Y_target.reshape(-1, 1)

m = len(Y_target)

# Add intercept term
X_b = np.c_[np.ones((m, 1)), X_feature]  # shape (100, 2)

# Initialize theta
theta = np.zeros((2, 1))

# Gradient Descent Settings
alpha = 0.01
iterations = 1500

# Cost function
def compute_cost(X, Y, theta):
    errors = X @ theta - Y
    return (1 / (2 * m)) * np.dot(errors.T, errors)

# Gradient descent
def gradient_descent(X, Y, theta, alpha, iterations):
    for _ in range(iterations):
        gradient = (1 / m) * X.T @ (X @ theta - Y)
        theta -= alpha * gradient
    return theta

# Train
theta = gradient_descent(X_b, Y_target, theta, alpha, iterations)

print("Learned theta:", theta)
