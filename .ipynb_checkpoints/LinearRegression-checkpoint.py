import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import seaborn as sns

class LinearRegressionClass:
    def __init__(self, data, target, learning_rate, epsilon, max_iterations= 1000, gd = False, cost_function_type = "L2") -> None:
         
        """
        @data: pandas dataframe or np.array, Passed data,

        @does: ,
        @return: ,
        """

        self.X = np.insert(data,0,1,axis=1)
        self.y = target
        self.cost_function_type = cost_function_type
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.gd = gd
        self.weight = None
    
    def rank(self):
        u, sigma, v = np.linalg.svd(self.X)
        return np.count_nonzero(sigma > 0)
    

    def linear_regression_closed_form(self):
        self.weight = np.matmul(np.linalg.inv(np.matmul(self.X.T , self.X)), np.matmul(self.X.T, self.y))
    

    def predict(self, X):
        return np.matmul(X, self.weight)

    def cost_functions(self, cost_type= "L2"):
        if cost_type == "L2":
            y_hat = self.predict(self.X)
            print(y_hat)
            print(self.y)
            return ((y_hat - self.y) ** 2).sum()


    def derivative_cost_functions(self, cost_type= "L2"):
        if cost_type == "L2":
            y_hat = self.predict(self.X)
            return 2* np.matmul(self.X.T, (y_hat - self.y))

    def gradient_descent(self, cost_type= "L2"):
        costs = []
        iterations = []
        previous_cost = math.inf
        for iter in tqdm(range(self.max_iterations)):

            step = self.learning_rate * self.derivative_cost_functions()
            self.weight = self.weight - step

            current_cost = self.cost_functions()
            costs.append(current_cost)
            iterations.append(iter)
            if abs(current_cost - previous_cost) < self.epsilon:
                print("Done")
                break
            previous_cost = current_cost
        self.fig_plot(x_data = costs, y_data = iterations)
    

    def fit(self):

        if self.X.shape[0] <1000 and not self.gd and self.rank() == self.X.shape[1] and (self.X.shape[0] > self.X.shape[1]):
            self.linear_regression_closed_form()
            print("Closed Form used")
            print(self.weight)
        else:
            self.weight = np.zeros(self.X.shape[1])
            self.gradient_descent()
            print("Gradient descent used")
            print(self.weight)

    def fig_plot(self, x_data, y_data):
        sns.lineplot(x= x_data, y= y_data)



