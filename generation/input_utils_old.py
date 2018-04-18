import os

import numpy as np


def load_mnist(path):
    fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int32)

    fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int32)
    # normalize to 0 1 float
    trX = trX / 255.
    teX = teX / 255.
    return trX, trY, teX, teY


def load_mmnist(path, samples_tr=200000, samples_te=10000):
    mnist = {}
    # train images
    trX = np.fromfile(file=os.path.join(path, 'trX'), dtype=np.uint8)
    mnist["trX"] = trX.reshape([samples_tr, 36, 36, 1]).astype(np.float32) / 255.
    # test images
    te0X = np.fromfile(file=os.path.join(path, 'te0X'), dtype=np.uint8)
    mnist["te0X"] = te0X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te1X = np.fromfile(file=os.path.join(path, 'te1X'), dtype=np.uint8)
    mnist["te1X"] = te1X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te2X = np.fromfile(file=os.path.join(path, 'te2X'), dtype=np.uint8)
    mnist["te2X"] = te2X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te3X = np.fromfile(file=os.path.join(path, 'te3X'), dtype=np.uint8)
    mnist["te3X"] = te3X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te4X = np.fromfile(file=os.path.join(path, 'te4X'), dtype=np.uint8)
    mnist["te4X"] = te4X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te5X = np.fromfile(file=os.path.join(path, 'te5X'), dtype=np.uint8)
    mnist["te5X"] = te5X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te6X = np.fromfile(file=os.path.join(path, 'te6X'), dtype=np.uint8)
    mnist["te6X"] = te6X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te7X = np.fromfile(file=os.path.join(path, 'te7X'), dtype=np.uint8)
    mnist["te7X"] = te7X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te8X = np.fromfile(file=os.path.join(path, 'te8X'), dtype=np.uint8)
    mnist["te8X"] = te8X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    teR30 = np.fromfile(file=os.path.join(path, 'teR30X'), dtype=np.uint8)
    mnist["teR30X"] = teR30.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    teR60 = np.fromfile(file=os.path.join(path, 'teR60X'), dtype=np.uint8)
    mnist["teR60X"] = teR60.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    teR30R = np.fromfile(file=os.path.join(path, 'teR30RX'), dtype=np.uint8)
    mnist["teR30RX"] = teR30R.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    teR60R = np.fromfile(file=os.path.join(path, 'teR60RX'), dtype=np.uint8)
    mnist["teR60RX"] = teR60R.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.

    # train labels
    trY = np.fromfile(file=os.path.join(path, 'trY'), dtype=np.int32)
    mnist["trY"] = trY.reshape([samples_tr, 2])
    # test labels
    te0Y = np.fromfile(file=os.path.join(path, 'te0Y'), dtype=np.int32)
    mnist["te0Y"] = te0Y.reshape([samples_te, 2])
    te1Y = np.fromfile(file=os.path.join(path, 'te1Y'), dtype=np.int32)
    mnist["te1Y"] = te1Y.reshape([samples_te, 2])
    te2Y = np.fromfile(file=os.path.join(path, 'te2Y'), dtype=np.int32)
    mnist["te2Y"] = te2Y.reshape([samples_te, 2])
    te3Y = np.fromfile(file=os.path.join(path, 'te3Y'), dtype=np.int32)
    mnist["te3Y"] = te3Y.reshape([samples_te, 2])
    te4Y = np.fromfile(file=os.path.join(path, 'te4Y'), dtype=np.int32)
    mnist["te4Y"] = te4Y.reshape([samples_te, 2])
    te5Y = np.fromfile(file=os.path.join(path, 'te5Y'), dtype=np.int32)
    mnist["te5Y"] = te5Y.reshape([samples_te, 2])
    te6Y = np.fromfile(file=os.path.join(path, 'te6Y'), dtype=np.int32)
    mnist["te6Y"] = te6Y.reshape([samples_te, 2])
    te7Y = np.fromfile(file=os.path.join(path, 'te7Y'), dtype=np.int32)
    mnist["te7Y"] = te7Y.reshape([samples_te, 2])
    te8Y = np.fromfile(file=os.path.join(path, 'te8Y'), dtype=np.int32)
    mnist["te8Y"] = te8Y.reshape([samples_te, 2])
    teR30 = np.fromfile(file=os.path.join(path, 'teR30Y'), dtype=np.int32)
    mnist["teR30Y"] = teR30.reshape([samples_te, 2])
    teR60 = np.fromfile(file=os.path.join(path, 'teR60Y'), dtype=np.int32)
    mnist["teR60Y"] = teR60.reshape([samples_te, 2])

    teR30R = np.fromfile(file=os.path.join(path, 'teR30RY'), dtype=np.int32)
    mnist["teR30RY"] = teR30R.reshape([samples_te, 2])
    teR60R = np.fromfile(file=os.path.join(path, 'teR60RY'), dtype=np.int32)
    mnist["teR60RY"] = teR60R.reshape([samples_te, 2])
    return mnist


def load_submmnist(path, samples_tr=200000, samples_te=10000):
    mnist = {}
    # train images
    trX = np.fromfile(file=os.path.join(path, 'trX'), dtype=np.uint8)
    mnist["trX"] = trX.reshape([samples_tr, 36, 36, 1]).astype(np.float32) / 255.
    # test images
    te0X = np.fromfile(file=os.path.join(path, 'te0X'), dtype=np.uint8)
    mnist["te0X"] = te0X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te1X = np.fromfile(file=os.path.join(path, 'te1X'), dtype=np.uint8)
    mnist["te1X"] = te1X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te2X = np.fromfile(file=os.path.join(path, 'te2X'), dtype=np.uint8)
    mnist["te2X"] = te2X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te3X = np.fromfile(file=os.path.join(path, 'te3X'), dtype=np.uint8)
    mnist["te3X"] = te3X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te4X = np.fromfile(file=os.path.join(path, 'te4X'), dtype=np.uint8)
    mnist["te4X"] = te4X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te5X = np.fromfile(file=os.path.join(path, 'te5X'), dtype=np.uint8)
    mnist["te5X"] = te5X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te6X = np.fromfile(file=os.path.join(path, 'te6X'), dtype=np.uint8)
    mnist["te6X"] = te6X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te7X = np.fromfile(file=os.path.join(path, 'te7X'), dtype=np.uint8)
    mnist["te7X"] = te7X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te8X = np.fromfile(file=os.path.join(path, 'te8X'), dtype=np.uint8)
    mnist["te8X"] = te8X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    # teR30 = np.fromfile(file=os.path.join(path, 'teR30X'), dtype=np.uint8)
    # mnist["teR30X"] = teR30.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    # teR60 = np.fromfile(file=os.path.join(path, 'teR60X'), dtype=np.uint8)
    # mnist["teR60X"] = teR60.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    teR30R = np.fromfile(file=os.path.join(path, 'teR30RX'), dtype=np.uint8)
    mnist["teR30RX"] = teR30R.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    teR60R = np.fromfile(file=os.path.join(path, 'teR60RX'), dtype=np.uint8)
    mnist["teR60RX"] = teR60R.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.

    # train labels
    trY = np.fromfile(file=os.path.join(path, 'trY'), dtype=np.int32)
    mnist["trY"] = trY.reshape([samples_tr, 2])
    # test labels
    te0Y = np.fromfile(file=os.path.join(path, 'te0Y'), dtype=np.int32)
    mnist["te0Y"] = te0Y.reshape([samples_te, 2])
    te1Y = np.fromfile(file=os.path.join(path, 'te1Y'), dtype=np.int32)
    mnist["te1Y"] = te1Y.reshape([samples_te, 2])
    te2Y = np.fromfile(file=os.path.join(path, 'te2Y'), dtype=np.int32)
    mnist["te2Y"] = te2Y.reshape([samples_te, 2])
    te3Y = np.fromfile(file=os.path.join(path, 'te3Y'), dtype=np.int32)
    mnist["te3Y"] = te3Y.reshape([samples_te, 2])
    te4Y = np.fromfile(file=os.path.join(path, 'te4Y'), dtype=np.int32)
    mnist["te4Y"] = te4Y.reshape([samples_te, 2])
    te5Y = np.fromfile(file=os.path.join(path, 'te5Y'), dtype=np.int32)
    mnist["te5Y"] = te5Y.reshape([samples_te, 2])
    te6Y = np.fromfile(file=os.path.join(path, 'te6Y'), dtype=np.int32)
    mnist["te6Y"] = te6Y.reshape([samples_te, 2])
    te7Y = np.fromfile(file=os.path.join(path, 'te7Y'), dtype=np.int32)
    mnist["te7Y"] = te7Y.reshape([samples_te, 2])
    te8Y = np.fromfile(file=os.path.join(path, 'te8Y'), dtype=np.int32)
    mnist["te8Y"] = te8Y.reshape([samples_te, 2])
    # teR30 = np.fromfile(file=os.path.join(path, 'teR30Y'), dtype=np.int32)
    # mnist["teR30Y"] = teR30.reshape([samples_te, 2])
    # teR60 = np.fromfile(file=os.path.join(path, 'teR60Y'), dtype=np.int32)
    # mnist["teR60Y"] = teR60.reshape([samples_te, 2])

    teR30R = np.fromfile(file=os.path.join(path, 'teR30RY'), dtype=np.int32)
    mnist["teR30RY"] = teR30R.reshape([samples_te, 2])
    teR60R = np.fromfile(file=os.path.join(path, 'teR60RY'), dtype=np.int32)
    mnist["teR60RY"] = teR60R.reshape([samples_te, 2])
    return mnist



def load_mmnist4(path, samples_tr=200000, samples_te=10000):
    mnist = {}
    # train images
    trX = np.fromfile(file=os.path.join(path, 'trX'), dtype=np.uint8)
    mnist["trX"] = trX.reshape([samples_tr, 36, 36, 1]).astype(np.float32) / 255.
    # test images
    te0X = np.fromfile(file=os.path.join(path, 'te0X'), dtype=np.uint8)
    mnist["te0X"] = te0X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te1X = np.fromfile(file=os.path.join(path, 'te1X'), dtype=np.uint8)
    mnist["te1X"] = te1X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te2X = np.fromfile(file=os.path.join(path, 'te2X'), dtype=np.uint8)
    mnist["te2X"] = te2X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te3X = np.fromfile(file=os.path.join(path, 'te3X'), dtype=np.uint8)
    mnist["te3X"] = te3X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te4X = np.fromfile(file=os.path.join(path, 'te4X'), dtype=np.uint8)
    mnist["te4X"] = te4X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te5X = np.fromfile(file=os.path.join(path, 'te5X'), dtype=np.uint8)
    mnist["te5X"] = te5X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te6X = np.fromfile(file=os.path.join(path, 'te6X'), dtype=np.uint8)
    mnist["te6X"] = te6X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te7X = np.fromfile(file=os.path.join(path, 'te7X'), dtype=np.uint8)
    mnist["te7X"] = te7X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te8X = np.fromfile(file=os.path.join(path, 'te8X'), dtype=np.uint8)
    mnist["te8X"] = te8X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.

    teR30R = np.fromfile(file=os.path.join(path, 'teR30RX'), dtype=np.uint8)
    mnist["teR30RX"] = teR30R.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    teR60R = np.fromfile(file=os.path.join(path, 'teR60RX'), dtype=np.uint8)
    mnist["teR60RX"] = teR60R.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.

    # train labels
    trY = np.fromfile(file=os.path.join(path, 'trY'), dtype=np.int32)
    mnist["trY"] = trY.reshape([samples_tr, 4])
    # test labels
    te0Y = np.fromfile(file=os.path.join(path, 'te0Y'), dtype=np.int32)
    mnist["te0Y"] = te0Y.reshape([samples_te, 4])
    te1Y = np.fromfile(file=os.path.join(path, 'te1Y'), dtype=np.int32)
    mnist["te1Y"] = te1Y.reshape([samples_te, 4])
    te2Y = np.fromfile(file=os.path.join(path, 'te2Y'), dtype=np.int32)
    mnist["te2Y"] = te2Y.reshape([samples_te, 4])
    te3Y = np.fromfile(file=os.path.join(path, 'te3Y'), dtype=np.int32)
    mnist["te3Y"] = te3Y.reshape([samples_te, 4])
    te4Y = np.fromfile(file=os.path.join(path, 'te4Y'), dtype=np.int32)
    mnist["te4Y"] = te4Y.reshape([samples_te, 4])
    te5Y = np.fromfile(file=os.path.join(path, 'te5Y'), dtype=np.int32)
    mnist["te5Y"] = te5Y.reshape([samples_te, 4])
    te6Y = np.fromfile(file=os.path.join(path, 'te6Y'), dtype=np.int32)
    mnist["te6Y"] = te6Y.reshape([samples_te, 4])
    te7Y = np.fromfile(file=os.path.join(path, 'te7Y'), dtype=np.int32)
    mnist["te7Y"] = te7Y.reshape([samples_te, 4])
    te8Y = np.fromfile(file=os.path.join(path, 'te8Y'), dtype=np.int32)
    mnist["te8Y"] = te8Y.reshape([samples_te, 4])

    teR30R = np.fromfile(file=os.path.join(path, 'teR30RY'), dtype=np.int32)
    mnist["teR30RY"] = teR30R.reshape([samples_te, 4])
    teR60R = np.fromfile(file=os.path.join(path, 'teR60RY'), dtype=np.int32)
    mnist["teR60RY"] = teR60R.reshape([samples_te, 4])
    return mnist
