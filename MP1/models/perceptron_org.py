"""Perceptron model."""

import numpy as np
from tqdm import tqdm


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        

        N,D = X_train.shape # (40000, 3072)


        if self.w is None:
            self.w = np.random.randn(D,self.n_class) # (3072,10)


        loss_hist = []
        
        for iter in tqdm(range(self.epochs)):
            loss = 0.0

            # compute the loss and the weight
            for i in range(N): # loop over 40,000 pics
                
                # (w_c.T) * x_i
                scores = np.dot(self.w.T, X_train[i]) 
                # (w_y.T) * x_i
                correct_class_score = scores[y_train[i]] 


                for idx_class in range(self.n_class):
                    
                    # if we got correct answer, do nothing
                    if idx_class == y_train[i]:
                        continue
                    # if not we need to compute gradient and update it
                    margin = scores[idx_class] - correct_class_score
                    
                    # apply hinge loss
                    max_margin = np.maximum(0,margin)
                    # print(max_margin)
                    # print("{} margin".format(max_margin))
                    if max_margin > 0:

                        loss += max_margin
                        # reinfore our decision

                        # penalize weight when its label is wrong
                        self.w[:,idx_class] = self.w[:,idx_class] - self.lr*X_train[i]

                        
                        # add weight when its label is correct
                        self.w[:,y_train[i]] = self.w[:,y_train[i]] + self.lr*X_train[i]

        loss /= N
        loss_hist.append(loss)
        self.w /= N
        
        print("{} epoch: {} loss".format(iter, loss))
        

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
        print("predicting begin")
        pred = []
        for test in X_test:
            predicted = np.argmax(np.dot(self.w.T, test))
            pred.append(predicted)
        return pred

