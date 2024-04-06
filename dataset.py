import numpy as np

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() # import mnist dataset

train_data = []
test_data = []

# merge x_train and y_train
for x, y in zip(x_train, y_train):
    x = np.array(x).astype(float).reshape(-1, 1)
    x /= 255 # normalize image

    res = np.zeros((10, 1))
    res[y] = 1

    data = (x, res) # x: array that represent number image, y: digit expected

    train_data.append(data)

# merge x_test and y_test
for x, y in zip(x_test, y_test):
    x = np.array(x).astype(float).reshape(-1, 1)
    x /= 255 
    
    data = (x, y)

    test_data.append(data)

