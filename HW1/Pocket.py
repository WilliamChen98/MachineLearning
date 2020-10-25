import numpy as np
import random


# 训练感知机模型
class Pocket:
    def __init__(self, input_samples, input_labels):
        self.samples = input_samples
        self.labels = input_labels
        self.weight = np.zeros((input_samples.shape[1], 1))
        self.threshold = 0
        self.best_weight = np.zeros((input_samples.shape[1], 1))
        self.best_threshold = 0
        self.sample_size = self.samples.shape[0]

    def sign(self, w, b, x):
        y = np.dot(x, w) + b
        return int(y)

    def update(self, sample_label, sample_data):
        tmp = sample_label * sample_data
        tmp = tmp.reshape(self.weight.shape)

        tmp_weight = tmp + self.weight
        tmp_threshold = self.threshold + sample_label
        if len(self.classify(self.best_weight, self.best_threshold)) >= len(self.classify(tmp_weight, tmp_threshold)):
            self.best_weight = tmp + self.weight
            self.best_threshold = self.threshold + sample_label
        self.weight = tmp + self.weight
        self.threshold = self.threshold + sample_label

    def classify(self, current_weight, current_threshold):
        mistakes = []
        for i in range(self.sample_size):
            tmp_y = self.sign(current_weight, current_threshold, self.samples[i, :])
            if tmp_y * self.labels[i] <= 0:
                mistakes.append(i)
        return mistakes

    def train(self, max_iteration_number):
        iteration_number = 0
        is_find = False
        while not is_find:
            mistakes = self.classify(self.weight, self.threshold)
            if len(mistakes) == 0:
                print('最终训练得到的w和b为：', self.best_weight, self.best_threshold)
                print('迭代次数为：', iteration_number)
                is_find = True
            n = mistakes[random.randint(0, len(mistakes) - 1)]
            self.update(self.labels[n], self.samples[n, :])
            iteration_number += 1
            if iteration_number == max_iteration_number:
                print('最终训练得到的w和b为：', self.best_weight, self.best_threshold)
                print('迭代次数为：', iteration_number)
                break
        if is_find:
            print('找到线性可分解。')
        return self.best_weight, self.best_threshold
