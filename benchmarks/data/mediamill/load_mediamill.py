import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def load_mediamill():
    x_train = np.zeros([30993, 120])
    x_test = np.zeros([12914, 120])
    y_train = []
    y_test = []

    for i in range(30993):
        y_train.append([])


    for ii in  range(12914):
        y_test.append([])

    with open("train-exp1.svm",'r') as f:
        steps = 0
        for line in f:
            #print(steps)
            #line = line.strip()
            line = line.replace("\n", "")
            tokens = line.split(" ")
            #第一项
            tokens_0 = tokens[0].split(",")
            #print(tokens)
            if tokens_0 == "":
                pass
            else:
                for t in tokens_0:
                    if t == '':
                        pass
                        #print("t", t, t=='')
                    else:
                        y_train[steps].append(int(t))
            #其余项
            for n in range(len(tokens) - 1):
                tmp = tokens[n+1]
                #print(tmp)
                tmp_pre, tmp_post = tmp.split(":")
                x_train[steps, int(tmp_pre)-1] = float(tmp_post)

            steps += 1

    with open("test-exp1.svm", "r") as f:
        steps = 0
        for line in f:
            # print(steps)
            # line = line.strip()
            line = line.replace("\n", "")
            tokens = line.split(" ")
            # 第一项
            tokens_0 = tokens[0].split(",")
            # print(tokens)
            if tokens_0 == "":
                pass
            else:
                for t in tokens_0:
                    if t == '':
                        pass
                        # print("t", t, t=='')
                    else:
                        y_test[steps].append(int(t))
            # 其余项
            for n in range(len(tokens) - 1):
                tmp = tokens[n + 1]
                # print(tmp)
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
    x_train, x_test, y_train, y_test = load_mediamill()
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)

    print(x_train[0])
    print(y_train[0])

