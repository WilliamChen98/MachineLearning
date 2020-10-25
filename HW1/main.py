import time
import pandas as pd
from printResult import Picture
from PLA import Perceptron
from Pocket import Pocket


def load_samples_and_labels():
    data1 = pd.read_csv("data2_2000.csv", header=None)
    sample = data1.iloc[:, :2].values
    label = data1.iloc[:, 2].values
    return sample, label


if __name__ == '__main__':
    start = time.time()
    samples, labels = load_samples_and_labels()
    my_perceptron = Perceptron(input_samples=samples, input_labels=labels)
    weights, bias = my_perceptron.train(100000)
    time_cost = time.time() - start
    print("PLA Time used:", time_cost)
    Picture = Picture(weights, bias, labels, 'Perceptron Learning Algorithm')
    Picture.show(samples)

    start = time.time()
    samples, labels = load_samples_and_labels()
    my_Pocket = Pocket(input_samples=samples, input_labels=labels)
    weights, bias = my_Pocket.train(100000)
    time_cost = time.time() - start
    print("Pocket Time used:", time_cost)
    Picture = Picture(weights, bias, labels, 'Pocket Algorithm')
    Picture.show(samples)
