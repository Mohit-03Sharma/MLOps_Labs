# Keras Tuner Lab — Hyperparameter Tuning with RandomSearch

## Overview
This lab demonstrates the process of hyperparameter tuning using Keras Tuner's 
RandomSearch algorithm. The goal is to automatically find the optimal set of 
hyperparameters for a neural network rather than manually guessing and checking values.

## Dataset
Two datasets were used in this lab:

**Boston Housing** — A regression dataset containing 404 training samples and 
102 test samples, each with 13 numerical features describing properties of homes 
in Boston. The target variable is the median house price in thousands of dollars.

**Fashion MNIST** — An image classification dataset containing 60,000 training 
samples and 10,000 test samples of 28x28 grayscale images across 10 clothing categories.
It was used as a validation experiment to test tuning effectiveness on a larger dataset.

## Model Architecture
Both models follow a fully-connected (Dense) neural network structure:
- A Dense layer with a tunable number of units and ReLU activation
- A Dropout layer (rate 0.2) to prevent overfitting
- A second Dense layer for further feature refinement
- An output layer — single neuron for regression (Boston Housing), 
  10 neurons with softmax for classification (Fashion MNIST)

## Hyperparameters Tuned
- **Units in the first Dense layer** — searched between 32 and 512
- **Learning rate for the Adam optimizer** — tested values: 0.01, 0.001, 0.0001

## Tuning Strategy
RandomSearch was used as the tuning strategy with 10 trials per experiment. 
It randomly samples hyperparameter combinations from the defined search space 
and evaluates each one against validation performance, returning the best 
configuration found across all trials.

## Key Findings
On the Boston Housing dataset, the baseline model slightly outperformed the 
hypertuned model, highlighting a known limitation of automated tuning on small 
datasets with too few samples, individual trials are not evaluated reliably 
enough for the tuner to make consistent recommendations.

On Fashion MNIST, the hypertuned model matched and marginally exceeded baseline 
accuracy while using significantly fewer parameters, demonstrating that on larger 
datasets, automated tuning reliably produces leaner and more efficient models.

## Requirements
- TensorFlow / Keras
- Keras Tuner (`pip install keras-tuner`)
- NumPy