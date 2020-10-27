import numpy as np


# pla algorithm
class Perceptron:
    def __init__(self, input_samples, input_labels):
        self.samples = input_samples
        self.labels = input_labels
        self.weight = np.zeros((input_samples.shape[1], 1))
        self.threshold = 0
        self.sample_size = self.samples.shape[0]

    def sign(self, w, b, x):
        y = np.dot(x, w) + b
        return int(y)

    def update(self, sample_label, sample_data):
        tmp = sample_label * sample_data
        tmp = tmp.reshape(self.weight.shape)
        self.weight = tmp + self.weight
        self.threshold = self.threshold + sample_label

    def train(self, max_iteration_number):
        is_find = False
        iteration_number = 0
        while not is_find:
            failed_point_number = 0
            iteration_number += 1
            for i in range(self.sample_size):
                tmp_y = self.sign(self.weight, self.threshold, self.samples[i, :])
                if tmp_y * self.labels[i] <= 0:  # if current point is wrong
                    failed_point_number += 1
                    self.update(self.labels[i], self.samples[i, :])
            if iteration_number == max_iteration_number:
                print('最终训练得到的w和b为：', self.weight, self.threshold)
                print('迭代次数为：', iteration_number)
                break
            if failed_point_number == 0:
                print('迭代次数为：', iteration_number)
                is_find = True
        return self.weight, self.threshold
