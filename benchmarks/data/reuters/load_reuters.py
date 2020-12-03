from tensorflow import keras


def load_reuters(vocab_size=10000, pad_length=128, num_classes=46):
    reuters = keras.datasets.reuters

    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=vocab_size)

    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=pad_length)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=0,
                                                       padding='post',
                                                       maxlen=pad_length)
    #train_labels = keras.utils.to_categorical(train_labels, num_classes)
    #test_labels = keras.utils.to_categorical(test_labels, num_classes)
    return train_data, test_data, train_labels, test_labels
