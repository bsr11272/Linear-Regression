import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import seaborn as sns

class LinearRegressionClass:
    def __init__(self, data, target, learning_rate = 0.00005, epsilon = 1e-8, 
                 max_iterations= 1000, gd = False, loss_function_type = "L2", huber_delta = 1, 
                 reg_type = None, reg_lambda = 0, print_log = False) -> None:
         
        """
        @data: pandas dataframe or np.array, Passed data,

        @does: ,
        @return: ,
        """
        
        self.X = np.insert(data.to_numpy(),0,1,axis=1)
        self.y = target.to_numpy()
        self.loss_function_type = loss_function_type
        self.reg_type = reg_type
        self.reg_lambda = 0 if self.reg_type == None else reg_lambda
        self.learning_rate = learning_rate
        self.huber_delta = huber_delta
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.gd = gd
        self.weight = None
        
        self.print_log = print_log
    
    def rank(self):
        u, sigma, v = np.linalg.svd(self.X)
        return np.count_nonzero(sigma > 0)
    

    def linear_regression_closed_form(self):
        print(self.X.T)
        print(np.matmul(self.X.T, self.y))
        print(np.linalg.inv(np.matmul(self.X.T , self.X)))
        self.weight = np.matmul(np.linalg.inv(np.matmul(self.X.T , self.X)), np.matmul(self.X.T, self.y))
    

    def predict(self, X):
        return np.matmul(X, self.weight)

    def loss_functions(self, X, y):
        y_hat = self.predict(X)
        if self.loss_function_type == "L2":
            return ((y_hat - y) ** 2).sum()
        
        if self.loss_function_type == "L1":
            return (abs(y_hat - y)).sum()
        
        if self.loss_function_type == "huber":
            huber_mse = 0.5*((y_hat - y) ** 2)
            huber_mae = self.huber_delta * (abs(y_hat - y) - 0.5 * self.huber_delta)
            return (np.where(np.abs(y - y_hat) <= self.huber_delta, huber_mse, huber_mae)).sum()


    def derivative_loss_functions(self, X, y):
        y_hat = self.predict(X)
        if self.loss_function_type == "L2":
            return 2* np.matmul(X.T, (y_hat - y))
        
        if self.loss_function_type == "L1":
            return np.matmul(X.T, np.sign(y_hat - y))
        
        if self.loss_function_type == "huber":
            return np.where(np.abs(y_hat - y) <= self.huber_delta, np.matmul(X.T, (y_hat - y)) , self.huber_delta * np.matmul(X.T, np.sign(y_hat - y)))
        
    def reg_functions(self):
        if self.reg_type == "L2":
            return ((self.weight) ** 2).sum()
        
        if self.reg_type == "L1":
            return (abs(self.weight)).sum()
        else:
            return 0

    def derivative_reg_functions(self):

        if self.reg_type == "L2":
            return 2* self.weight
        
        if self.reg_type == "L1":
            return np.sign(self.weight)
        
        else:
            return 0
        
    def gradient_descent(self, X, y):
        costs = []
        iterations = []
        previous_cost = math.inf
        for iter in tqdm(range(1, self.max_iterations)):
            
            step = self.learning_rate * (self.derivative_loss_functions(X, y) + self.reg_lambda * self.derivative_reg_functions())
            
            self.weight = self.weight - step
            
            current_cost = self.loss_functions(X, y) + self.reg_lambda * self.reg_functions()
            #print(current_cost)
            costs.append(current_cost)
            iterations.append(iter)
            
            if self.print_log:
                print("Step : " + str(iter) + ": ","\n")
                print("\t" + "change ",step, "\n")
                print("current_weights: ", self.weight)
                print("current_cost: ", current_cost)
                print("current_cost_change: ", abs(current_cost - previous_cost), "\n")
            
            if abs(current_cost - previous_cost) < self.epsilon:
                print("Done")
                break
            previous_cost = current_cost
        self.fig_plot(x_data = iterations, y_data = costs)
    

    def fit(self):

        if self.X.shape[0] <1000 and not self.gd and (self.rank() == self.X.shape[1]) and (self.X.shape[0] > self.X.shape[1]):
            self.linear_regression_closed_form()
            print("Closed Form used")
            print(self.weight)
        else:            
            #self.weight = np.random.randn(self.X.shape[1])
            self.weight = np.zeros(self.X.shape[1])
            temp = np.matmul(self.X.T, self.X)
            print(temp, "\n")
            print(np.conj(temp).T, "\n")
            self.gradient_descent(self.X, self.y)
            print("Gradient descent used")
            print(self.weight)

    def fig_plot(self, x_data, y_data):
        sns.lineplot(x= x_data, y= y_data)


