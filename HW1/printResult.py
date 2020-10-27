import numpy as np
import matplotlib.pyplot as plt


#  show the image of dataset and the separating line
class Picture:
    def __init__(self, w, b, labels, title):
        self.b = b
        self.w = w
        self.label = labels
        self.title = title
        plt.figure(1)
        plt.title(title, size=14)
        plt.xlabel('x0-axis', size=10)
        plt.ylabel('x1-axis', size=10)

    def show(self, data):
        x_data = np.linspace(0, 10, 100)
        y_data = (-self.b - self.w[0] * x_data) / self.w[1]
        plt.plot(x_data, y_data, color='b', label='sample data')
        for i in range(0, len(self.label)):
            if self.label[i] > 0:
                plt.scatter(data[i][0], data[i][1], color='r', s=50)
            else:
                plt.scatter(data[i][0], data[i][1], color='g', s=50, marker='x')
        plt.savefig(self.title + '_not linearly separable.png', dpi=75)
        plt.show()
