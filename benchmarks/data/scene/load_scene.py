import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def load_scene():
    x_train = np.zeros([1211, 294])
    x_test = np.zeros([1196, 294])
    y_train = []
    y_test = []

    for i in range(1211):
        y_train.append([])


    for ii in  range(1196):
        y_test.append([])

    with open("scene_train",'r') as f:
        steps = 0
        for line in f:
            tokens = line.split(" ")
            #第一项
            tokens_0 = tokens[0].split(",")
            #print(tokens)
            for t in tokens_0:
                y_train[steps].append(int(t))
            #其余项
            for n in range(len(tokens) - 1):
                tmp = tokens[n+1]
                #print(tmp)
                tmp_pre, tmp_post = tmp.split(":")
                x_train[steps, int(tmp_pre)-1] = float(tmp_post)

            steps += 1

    with open("scene_test", "r") as f:
        steps = 0
        for line in f:
            tokens = line.split(" ")
            # 第一项
            tokens_0 = tokens[0].split(",")
            for t in tokens_0:
                y_test[steps].append(int(t))
            # 其余项
            for n in range(len(tokens) - 1):
                tmp = tokens[n + 1]
                tmp_pre, tmp_post = tmp.split(":")
                x_test[steps, int(tmp_pre)-1] = float(tmp_post)

            steps += 1

    mb = MultiLabelBinarizer()
    y_train = mb.fit_transform(y_train)
    y_test = mb.fit_transform(y_test)
    # print("x_train", x_train)
    # print("y_train", y_train)
    # print("x_test", x_test)
    # print("y_test", y_test)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_scene()
    print(x_train.shape)
    print(y_train.shape)

    print(x_train[0])
    print(y_train[0])
