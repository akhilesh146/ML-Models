import numpy as np

class LogisticRegression:
  # constructor
  def __init__(self, learning_rate=0.001, n_iters=1000):
    self.learning_rate = learning_rate
    self.no_of_iterations = n_iters



  def fit(self, X, Y):
    # no of data points in the dataset (no of rows) --> m
    # no of input features in the dataset (no of columns) --> n
    self.m, self.n = X.shape
    
    self.w = np.zeros(self.n)
    self.b = 0
    self.X = X
    self.Y = Y

    for i in range(self.no_of_iterations):
      self.update_weights()



  def update_weights(self):
    Y_hat = 1 / (1 + np.exp(-(self.X.dot(self.w) + self.b)))

    #derivatives
    dw = (1/self.m) * np.dot(self.X.T, (Y_hat - self.Y))
    db = (1/self.m) * np.sum(Y_hat - self.Y)

    self.w = self.w - self.learning_rate * dw
    self.b = self.b - self.learning_rate * db


  # sigmoid equation and prediction
  def predict(self, X ):

    Y_pred = 1 / (1 + np.exp(-(self.X.dot(self.w) + self.b)))
    Y_pred = np.where(Y_pred > 0.5, 1, 0)
    return Y_pred
