# Assignment 1: Linear Classifiers 

In this assignment, implement simple linear classifiers and run them on two different datasets:

- Mushroom dataset: a simple categorical binary classification dataset. Please note that the labels in the dataset are 0/1, as opposed to -1/1 as in the lectures, so you may have to change either the labels or the derivations of parameter update rules accordingly.

- CIFAR-10: a multi-class image classification dataset

The goal of this assignment is to help you understand the fundamentals of a few classic methods and become familiar with scientific computing tools in Python. You will also get experience in hyperparameter tuning and using proper train/validation/test data splits.

I implemented the following classifiers (in their respective files):

1. Logistic regression (logistic.py)
2. Perceptron (perceptron.py)
3. SVM (svm.py)
4. Softmax (softmax.py)

For the logistic regression classifier, multi-class prediction is difficult, as it requires a 1v1 or 1vRest classifier for every class. Therefore, you only need to use logistic regression on the Mushroom dataset.
