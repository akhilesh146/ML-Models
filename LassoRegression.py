class LassoRegression():
  def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations
    self.lambda_parameter = lambda_parameter



  # fit function to train the model with dataset
  def fit(self, X, Y):
    self.m, self.n = X.shape
    self.X = X
    self.Y = Y

    self.w = np.zeros(self.n)
    self.b = 0

    # self.X_with_ones = np.hstack((self.X, np.ones((self.m, 1)))) 

    for i in range(self.no_of_iterations):
      self.update_weights()
  


  # function for updating weights
  def update_weights(self):
    # linear equation of model
    Y_prediction = self.predict(self.X)

    dw = np.zeros(self.n)

    # gradient (dw, db)
    for i in range(self.n):
      if self.w[i] > 0:
        dw[i] = -2*( (self.X[:, i].dot(self.Y - Y_prediction)) + self.lambda_parameter) / self.m
      else:
        dw[i] = -2*( (self.X[:, i].dot(self.Y - Y_prediction)) + self.lambda_parameter) / self.m

    db = -2 * np.sum(self.Y - Y_prediction) / self.m

    self.w = self.w - self.learning_rate * dw
    self.b = self.b - self.learning_rate * db



  # predict the target variable
  def predict(self, X):
    return X.dot(self.w) + self.b

  