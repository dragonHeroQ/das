import numpy as np
#from sklearn.preprocessing import MultiLabelBinarizer

def load_a9a():
    x_train = np.zeros([32561, 123])
    x_test = np.zeros([16281, 123])
    y_train = []
    y_test = []



    with open("a9a.train",'r') as f:
        steps = 0
        for line in f:
            tokens = line.strip().split(" ")
            #print("tokens:", tokens)
            #第一项
            tokens_0 = int(tokens[0])

            y_train.append(tokens_0)

            #其余项
            for n in range(len(tokens) - 1):
                tmp = tokens[n+1]
                #print(tmp)
                tmp_pre, tmp_post = tmp.split(":")
                x_train[steps, int(tmp_pre)-1] = float(tmp_post)

            steps += 1

    with open("a9a.t", "r") as f:
        steps = 0
        for line in f:
            tokens = line.strip().split(" ")
            #第一项
            tokens_0 = int(tokens[0])

            y_test.append(tokens_0)
            # 其余项
            for n in range(len(tokens) - 1):
                tmp = tokens[n + 1]
                tmp_pre, tmp_post = tmp.split(":")
                x_test[steps, int(tmp_pre)-1] = float(tmp_post)

            steps += 1

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_a9a()
    print(x_train.shape)
    print(y_train.shape)

    print(x_train[0])
    print(y_train[0])
