# main logistic regression class

import numpy as np


class LogisticRegression():

    def __init__(self, learning_rate, reg_power, l1_ratio, solver, num_iter):

        self.learning_rate = learning_rate
        self.reg_power = reg_power
        self.l1_ratio = l1_ratio
        self.solver = solver
        self.num_iter = num_iter

        self.params = None

    def _init_params(self, len_w):
        """
        Initialises the parameters to zeros
        """

        self.params = {'w': np.zeros(shape=(len_w, 1)), 'b': np.zeros(shape=(1, 1))}

    def _forward_pass(self, X):
        """
        Make a forward pass (or, evaluate) the inputs using current values of params
        Assumes that parameters are initialized already
        shape of X: (n_features, n_examples)
        """
        z = np.dot(np.transpose(self.params['w']), X) + self.params['b']
        return LogisticRegression.sigmoid(z)

    def _eval_cost(self, A, Y):
        """
        Evaluates totol cost of one forward pass, needs activations list A and true labels Y
        """

        _log_loss = LogisticRegression.log_loss(a=A, y=Y)
        _cost = _log_loss + self.reg_power * (
                self.l1_ratio * (np.sum(np.abs(self.params['w'])) + np.squeeze(np.abs(self.params['b']))) +
                ((1 - self.l1_ratio) / 2.0) * (np.sum(np.square(self.params['w'])) +
                                               np.squeeze(np.square(self.params['b']))))
        return _cost

    def fit(self, X, Y):
        """
        Fits logistic regression model parameters as per given hyperparameter choices
        X: Train data, shape = (n_features, n_examples)
        Y: Train labels, shape = (1, n_examples)
        """

        m = np.shape(X)[1]
        num_features = np.shape(X)[0]

        self._init_params(len_w = num_features)

        if self.solver == 'gradient-descent':
            self._solver_gradient_descent(X, Y)
        elif self.solver == 'newtons-method':
            self._solver_newtons_method(X, Y)

    def predict(self):
        pass

    def _solver_gradient_descent(self, X, Y):

        m = np.shape(X)[1]

        for i in range(self.num_iter):

            # Do forward pass, get activations
            _forward_pass_output = self._forward_pass(X)

            # Calculate total cost at the iteration
            _cost = self._eval_cost(A=_forward_pass_output, Y=Y)

            # Calculate gradients
            _dw = (1.0/m)*(np.dot(X, np.transpose(_forward_pass_output - Y))) + self.reg_power*(
                self.l1_ratio*(np.sign(self.params['w'])) + (1-self.l1_ratio)*(self.params['w'])
            )
            _db = (1.0/m)*(np.sum(_forward_pass_output - Y)) + self.reg_power*(
                self.l1_ratio*(np.sign(self.params['b'])) + (1-self.l1_ratio)*(self.params['b'])
            )

            # Update params
            self.params['w'] = self.params['w'] - self.learning_rate*_dw
            self.params['b'] = self.params['b'] - self.learning_rate*_db

    def _solver_gradient_descent_with_momentum(self, X, Y):

        m = np.shape(X)[1]
        p = np.shape(X)[0]
        beta = 0.9
        v_dw = np.zeros(shape= (p, 1))
        v_db = np.zeros(shape= (1, 1))

        for i in range(self.num_iter):

            # Do forward pass, get activations
            _forward_pass_output = self._forward_pass(X)

            # Calculate total cost at the iteration
            _cost = self._eval_cost(A=_forward_pass_output, Y=Y)

            # Calculate gradients
            _dw = (1.0/m)*(np.dot(X, np.transpose(_forward_pass_output - Y))) + self.reg_power*(
                self.l1_ratio*(np.sign(self.params['w'])) + (1-self.l1_ratio)*(self.params['w'])
            )
            _db = (1.0/m)*(np.sum(_forward_pass_output - Y)) + self.reg_power*(
                self.l1_ratio*(np.sign(self.params['b'])) + (1-self.l1_ratio)*(self.params['b'])
            )

            # calculate the momentum terms, TODO: implement error correction?
            v_dw = beta*v_dw + (1-beta)*_dw
            v_db = beta*v_db + (1-beta)*_db

            # Update params
            self.params['w'] = self.params['w'] - self.learning_rate*v_dw
            self.params['b'] = self.params['b'] - self.learning_rate*v_db

    def _solver_gradient_descent_with_rmsprop(self, X, Y):

        m = np.shape(X)[1]
        p = np.shape(X)[0]
        beta = 0.9
        epsilon = 1e-6

        s_dw = np.zeros(shape= (p, 1))
        s_db = np.zeros(shape= (1, 1))

        for i in range(self.num_iter):

            # Do forward pass, get activations
            _forward_pass_output = self._forward_pass(X)

            # Calculate total cost at the iteration
            _cost = self._eval_cost(A=_forward_pass_output, Y=Y)

            # Calculate gradients
            _dw = (1.0/m)*(np.dot(X, np.transpose(_forward_pass_output - Y))) + self.reg_power*(
                self.l1_ratio*(np.sign(self.params['w'])) + (1-self.l1_ratio)*(self.params['w'])
            )
            _db = (1.0/m)*(np.sum(_forward_pass_output - Y)) + self.reg_power*(
                self.l1_ratio*(np.sign(self.params['b'])) + (1-self.l1_ratio)*(self.params['b'])
            )

            # calculate the rms prop terms, TODO: implement error correction?
            s_dw = beta*s_dw + (1-beta)*np.multiply(_dw, _dw)
            s_db = beta*s_db + (1-beta)*np.multiply(_db, _db)

            # Update params
            self.params['w'] = self.params['w'] - np.divide((self.learning_rate*_dw),
                                                            (np.sqrt(s_dw) + epsilon))
            self.params['b'] = self.params['b'] - np.divide((self.learning_rate*_dw),
                                                            (np.sqrt(s_db) + epsilon))

    def _solver_gradient_descent_with_adam(self, X, Y):

        m = np.shape(X)[1]
        p = np.shape(X)[0]
        beta_v = 0.9
        beta_s = 0.9
        epsilon = 1e-6

        v_dw = np.zeros(shape=(p, 1))
        v_db = np.zeros(shape=(1, 1))

        s_dw = np.zeros(shape= (p, 1))
        s_db = np.zeros(shape= (1, 1))

        for i in range(self.num_iter):

            # Do forward pass, get activations
            _forward_pass_output = self._forward_pass(X)

            # Calculate total cost at the iteration
            _cost = self._eval_cost(A=_forward_pass_output, Y=Y)

            # Calculate gradients
            _dw = (1.0/m)*(np.dot(X, np.transpose(_forward_pass_output - Y))) + self.reg_power*(
                self.l1_ratio*(np.sign(self.params['w'])) + (1-self.l1_ratio)*(self.params['w'])
            )
            _db = (1.0/m)*(np.sum(_forward_pass_output - Y)) + self.reg_power*(
                self.l1_ratio*(np.sign(self.params['b'])) + (1-self.l1_ratio)*(self.params['b'])
            )

            # calculate the momentum and rms prop terms, TODO: implement error correction?
            v_dw = beta_v * v_dw + (1 - beta_v) * _dw
            v_db = beta_v * v_db + (1 - beta_v) * _db

            s_dw = beta_s*s_dw + (1-beta_s)*np.multiply(_dw, _dw)
            s_db = beta_s*s_db + (1-beta_s)*np.multiply(_db, _db)

            # Update params
            self.params['w'] = self.params['w'] - np.divide((self.learning_rate*v_dw),
                                                            (np.sqrt(s_dw) + epsilon))
            self.params['b'] = self.params['b'] - np.divide((self.learning_rate*v_dw),
                                                            (np.sqrt(s_db) + epsilon))

    # TODO: implement adagrad and adadelta?
    # TODO: usage of momentum, adam, adagrad in cases of minibatch gradient descent/sgd only

    def _solver_newtons_method(self, X, Y):
        pass

    @staticmethod
    def sigmoid(x):
        """
        Calculates sigmoid function for given input, x
        """
        return 1 / (1 + np.exp(-1 * np.array(x)))

    @staticmethod
    def log_loss(a, y):
        """
        Calculates the log loss for given true labels --> y and predictions --> a
        shape of a,y: (1, n_examples)
        """
        m = np.shape(a)[1]
        logloss = (-1 / m) * (np.dot(y, np.transpose(np.log(a))) +
                              np.dot((1 - y), np.transpose(np.log(1 - a))))
        return np.squeeze(logloss)    # logloss has shape (1, 1), getting float value using squeeze

