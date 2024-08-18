import numpy as np

class SVM_classifier():
  def __init__(self, learning_rate=0.001, n_iters=1000, lambda_param=0.01):
    self.learning_rate = learning_rate
    self.lambda_param = lambda_param
    self.n_iters = n_iters

  def fit(self, X, Y):
    self.X = X
    self.Y = Y
    self.m, self.n = X.shape

    self.w = np.zeros(self.n)
    self.b = 0

    for i in range(self.n_iters):
      self.update_weights()

  def update_weights(self):

    y_label = np.where(self.Y <= 0, -1, 1)

    for index, x_i in enumerate(self.X):
      condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1

      if condition:
        dw = 2 * self.lambda_param * self.w
        db = 0
      else:
        dw = 2 * self.lambda_param * self.w - np.dot(x_i, y_label[index])
        db = y_label[index]

      self.w = self.w - self.learning_rate * dw
      self.b = self.b - self.learning_rate * db

  def predict(self):

    output = np.dot(self.X, self.w) - self.b
    predicted_labels = np.sign(output)

    y_hat = np.where(predicted_labels <= -1, 0, 1)

    return y_hat
