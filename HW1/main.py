#  All these program were wrote by Chen Fangyue -- XS20033183


import time
import pandas as pd
from printResult import Picture
from PLA import Perceptron
from Pocket import Pocket


def load_samples_and_labels():
    # data = pd.read_csv("Separable_20.csv", header=None)
    data = pd.read_csv("not_Separable_20.csv", header=None)
    # data = pd.read_csv("Separable_2000.csv", header=None)
    # data = pd.read_csv("not_Separable_2000_10.csv", header=None)
    # data = pd.read_csv("not_Separable_2000_1000.csv", header=None)
    sample = data.iloc[:, :2].values
    label = data.iloc[:, 2].values
    return sample, label


if __name__ == '__main__':
    max_iteration_time = 1000  # change this when running inseparable dataset

    start = time.time()
    samples, labels = load_samples_and_labels()
    my_perceptron = Perceptron(input_samples=samples, input_labels=labels)
    weights, bias = my_perceptron.train(max_iteration_time)
    time_cost = time.time() - start
    print("PLA Time used:", time_cost)
    Picture_pla = Picture(weights, bias, labels, 'Perceptron Learning Algorithm')
    Picture_pla.show(samples)

    start = time.time()
    samples, labels = load_samples_and_labels()
    my_Pocket = Pocket(input_samples=samples, input_labels=labels)
    weights, bias = my_Pocket.train(max_iteration_time)
    time_cost = time.time() - start
    print("Pocket Time used:", time_cost)
    Picture_pocket = Picture(weights, bias, labels, 'Pocket Algorithm')
    Picture_pocket.show(samples)
