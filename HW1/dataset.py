import random
import pandas as pd

# 生成线性可分数据集
dataset1 = pd.DataFrame(columns=["X", "Y", "label"])
dataset2 = pd.DataFrame(columns=["X", "Y", "label"])
for i in range(0, 2000):
    X1 = random.uniform(0.0, 10.0)
    Y1 = random.uniform(0.0, 10.0)
    if X1 + Y1 - 10 > 0:
        label = 1
    else:
        label = -1
    dataset1 = dataset1.append(pd.DataFrame({"X": X1, "Y": Y1, "label": label}, columns=["X", "Y", "label"], index=[0]),
                               ignore_index=True)
dataset1.to_csv('data1_2000.csv', index=False, header=False)

# 生成线性不可分数据集
for i in range(0, 2000):
    X1 = random.uniform(0.0, 10.0)
    Y1 = random.uniform(0.0, 10.0)
    if X1 + Y1 - 10 > 0 and i < 1000:
        label = 1
    elif i < 1000:
        label = -1
    else:
        j = random.randint(0, 10)
        if j < 5:
            label = -1
        else:
            label = 1
    dataset2 = dataset2.append(pd.DataFrame({"X": X1, "Y": Y1, "label": label}, columns=["X", "Y", "label"], index=[0]),
                               ignore_index=True)
dataset2.to_csv('data2_2000.csv', index=False, header=False)
