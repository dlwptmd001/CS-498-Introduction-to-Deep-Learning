"""Logistic regression model."""

import numpy as np
import random

class Logistic:
    def __init__(self, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        # self.w = None  # TODO: change this
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.threshold = 0.5

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        return 1 / (1 + np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        print("start training")
        N, D = X_train.shape  # (num_examples, num_features)
        self.w = np.random.rand(D, )  # (22, 1)
        arr = list(range(N))
        print("initial weights => ", self.w)

        for epoch in range(self.epochs):
            random.shuffle(arr)
            for i in arr:
                curr_example = X_train[i]
                # curr_label = y_train[i]
                if y_train[i] == 1:
                    curr_label = 1
                else:
                    curr_label = -1

                y_hat = self.sigmoid(np.dot(curr_example, self.w))  # (1, 0)
                # loss = loss + y_hat - curr_label
                dw = self.sigmoid(-curr_label * np.dot(curr_example, self.w)) * curr_label * curr_example

                assert(dw.shape == self.w.shape)

                self.w = self.w + self.lr * dw

            # print("{}th epoch, loss = {}".format(epoch, loss))

        print("end training")
        # for epoch in range(self.epochs):
        #     z = np.dot(X_train, self.w)  # (4874, 22) * (22, 1) = (4874, 1)
        #     assert(z.shape == (4874, 1))
        #     print("before sigmoid z shape => ",z.shape)
        #     y_hat = self.sigmoid(z)  # (4874, 1)
        #     print("after sigmoid y_hat shape => ", y_hat.shape)
        #     loss = y_hat - y_train.reshape(N, 1)  # (4874, 1)
        #     print("loss shape => ", loss.shape)
        #     assert(loss.shape == (4874, 1))
        #     gradient = np.dot(X_train.T, loss)  # (22, 4874) * (4874, 1) = (22, 1)

        #     print("gradient shape = ", gradient.shape)
        #     assert(gradient.shape == (22, 1))

        #     dw = (2 / N) * gradient.sum(axis=1).reshape(D, 1)
        #     if epoch == 0:
        #         print("dw shape => ", dw.shape)

        #     self.w -= self.lr * dw

        #     print("{}th epoch, loss = {}".format(epoch, loss.sum()))
        #     print("{}th epoch, current weights = {}".format(epoch, self.w))

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        # out = np.zeros(X_test.shape[0])
        out = np.dot(X_test, self.w)
        out = np.where(self.sigmoid(out) >= 0.5, 1, 0)
        # print(out[:10])
        return out
