import os,time
# import scipy
import numpy as np
import tensorflow as tf
import collections

from myconfig import cfg


import os, time
# import scipy
import numpy as np
import tensorflow as tf
import collections

from myconfig import cfg


def load_mnist(path=cfg.dataset):
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

    # Training images
    trX = np.fromfile(file=os.path.join(path, 'trX'), dtype=np.uint8)
    mnist["trX"] = trX.reshape([samples_tr, 36, 36, 1]).astype(np.float32) / 255.


    # Test images--undercomplete dataset
    te0X = np.fromfile(file=os.path.join(path, 'tes0X'), dtype=np.uint8)
    mnist["tes0X"] = te0X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te1X = np.fromfile(file=os.path.join(path, 'tes1X'), dtype=np.uint8)
    mnist["tes1X"] = te1X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te2X = np.fromfile(file=os.path.join(path, 'tes2X'), dtype=np.uint8)
    mnist["tes2X"] = te2X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te3X = np.fromfile(file=os.path.join(path, 'tes3X'), dtype=np.uint8)
    mnist["tes3X"] = te3X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te4X = np.fromfile(file=os.path.join(path, 'tes4X'), dtype=np.uint8)
    mnist["tes4X"] = te4X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te5X = np.fromfile(file=os.path.join(path, 'tes5X'), dtype=np.uint8)
    mnist["tes5X"] = te5X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te6X = np.fromfile(file=os.path.join(path, 'tes6X'), dtype=np.uint8)
    mnist["tes6X"] = te6X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te7X = np.fromfile(file=os.path.join(path, 'tes7X'), dtype=np.uint8)
    mnist["tes7X"] = te7X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te8X = np.fromfile(file=os.path.join(path, 'tes8X'), dtype=np.uint8)
    mnist["tes8X"] = te8X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    teR30R = np.fromfile(file=os.path.join(path, 'tesR30RX'), dtype=np.uint8)
    mnist["tesR30RX"] = teR30R.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    teR60R = np.fromfile(file=os.path.join(path, 'tesR60RX'), dtype=np.uint8)
    mnist["tesR60RX"] = teR60R.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.

    # Test images--full dataset
    te0X = np.fromfile(file=os.path.join(path, 'tef0X'), dtype=np.uint8)
    mnist["tef0X"] = te0X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te1X = np.fromfile(file=os.path.join(path, 'tef1X'), dtype=np.uint8)
    mnist["tef1X"] = te1X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te2X = np.fromfile(file=os.path.join(path, 'tef2X'), dtype=np.uint8)
    mnist["tef2X"] = te2X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te3X = np.fromfile(file=os.path.join(path, 'tef3X'), dtype=np.uint8)
    mnist["tef3X"] = te3X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te4X = np.fromfile(file=os.path.join(path, 'tef4X'), dtype=np.uint8)
    mnist["tef4X"] = te4X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te5X = np.fromfile(file=os.path.join(path, 'tef5X'), dtype=np.uint8)
    mnist["tef5X"] = te5X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te6X = np.fromfile(file=os.path.join(path, 'tef6X'), dtype=np.uint8)
    mnist["tef6X"] = te6X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te7X = np.fromfile(file=os.path.join(path, 'tef7X'), dtype=np.uint8)
    mnist["tef7X"] = te7X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    te8X = np.fromfile(file=os.path.join(path, 'tef8X'), dtype=np.uint8)
    mnist["tef8X"] = te8X.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    teR30R = np.fromfile(file=os.path.join(path, 'tefR30RX'), dtype=np.uint8)
    mnist["tefR30RX"] = teR30R.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.
    teR60R = np.fromfile(file=os.path.join(path, 'tefR60RX'), dtype=np.uint8)
    mnist["tefR60RX"] = teR60R.reshape([samples_te, 36, 36, 1]).astype(np.float32) / 255.

    # Training labels labels
    trY = np.fromfile(file=os.path.join(path, 'trY'), dtype=np.int32)
    mnist["trY"] = trY.reshape([samples_tr, 2])


    # Test labels--undercomplete dataset
    te0Y = np.fromfile(file=os.path.join(path, 'tes0Y'), dtype=np.int32)
    mnist["tes0Y"] = te0Y.reshape([samples_te, 2])
    te1Y = np.fromfile(file=os.path.join(path, 'tes1Y'), dtype=np.int32)
    mnist["tes1Y"] = te1Y.reshape([samples_te, 2])
    te2Y = np.fromfile(file=os.path.join(path, 'tes2Y'), dtype=np.int32)
    mnist["tes2Y"] = te2Y.reshape([samples_te, 2])
    te3Y = np.fromfile(file=os.path.join(path, 'tes3Y'), dtype=np.int32)
    mnist["tes3Y"] = te3Y.reshape([samples_te, 2])
    te4Y = np.fromfile(file=os.path.join(path, 'tes4Y'), dtype=np.int32)
    mnist["tes4Y"] = te4Y.reshape([samples_te, 2])
    te5Y = np.fromfile(file=os.path.join(path, 'tes5Y'), dtype=np.int32)
    mnist["tes5Y"] = te5Y.reshape([samples_te, 2])
    te6Y = np.fromfile(file=os.path.join(path, 'tes6Y'), dtype=np.int32)
    mnist["tes6Y"] = te6Y.reshape([samples_te, 2])
    te7Y = np.fromfile(file=os.path.join(path, 'tes7Y'), dtype=np.int32)
    mnist["tes7Y"] = te7Y.reshape([samples_te, 2])
    te8Y = np.fromfile(file=os.path.join(path, 'tes8Y'), dtype=np.int32)
    mnist["tes8Y"] = te8Y.reshape([samples_te, 2])
    teR30R = np.fromfile(file=os.path.join(path, 'tesR30RY'), dtype=np.int32)
    mnist["tesR30RY"] = teR30R.reshape([samples_te, 2])
    teR60R = np.fromfile(file=os.path.join(path, 'tesR60RY'), dtype=np.int32)
    mnist["tesR60RY"] = teR60R.reshape([samples_te, 2])


    # Test labels--full dataset
    te0Y = np.fromfile(file=os.path.join(path, 'tef0Y'), dtype=np.int32)
    mnist["tef0Y"] = te0Y.reshape([samples_te, 2])
    te1Y = np.fromfile(file=os.path.join(path, 'tef1Y'), dtype=np.int32)
    mnist["tef1Y"] = te1Y.reshape([samples_te, 2])
    te2Y = np.fromfile(file=os.path.join(path, 'tef2Y'), dtype=np.int32)
    mnist["tef2Y"] = te2Y.reshape([samples_te, 2])
    te3Y = np.fromfile(file=os.path.join(path, 'tef3Y'), dtype=np.int32)
    mnist["tef3Y"] = te3Y.reshape([samples_te, 2])
    te4Y = np.fromfile(file=os.path.join(path, 'tef4Y'), dtype=np.int32)
    mnist["tef4Y"] = te4Y.reshape([samples_te, 2])
    te5Y = np.fromfile(file=os.path.join(path, 'tef5Y'), dtype=np.int32)
    mnist["tef5Y"] = te5Y.reshape([samples_te, 2])
    te6Y = np.fromfile(file=os.path.join(path, 'tef6Y'), dtype=np.int32)
    mnist["tef6Y"] = te6Y.reshape([samples_te, 2])
    te7Y = np.fromfile(file=os.path.join(path, 'tef7Y'), dtype=np.int32)
    mnist["tef7Y"] = te7Y.reshape([samples_te, 2])
    te8Y = np.fromfile(file=os.path.join(path, 'tef8Y'), dtype=np.int32)
    mnist["tef8Y"] = te8Y.reshape([samples_te, 2])
    teR30R = np.fromfile(file=os.path.join(path, 'tefR30RY'), dtype=np.int32)
    mnist["tefR30RY"] = teR30R.reshape([samples_te, 2])
    teR60R = np.fromfile(file=os.path.join(path, 'tefR60RY'), dtype=np.int32)
    mnist["tefR60RY"] = teR60R.reshape([samples_te, 2])


    return mnist


def load_submmnist(path, samples_tr=200000, samples_te=10000):
    """
    @Deprecated
    This method is now equivalent to load_mmnist. Exists for compatibility, but will be removed in later versions of TL.
    """
    return load_mmnist(path, samples_tr=samples_tr, samples_te=samples_te)



class TrainingMonitor:
    def __init__(self):
        self._hist_records = collections.OrderedDict()
    def add(self,name,value,num=20):
        if not name in self._hist_records:
            self._hist_records[name] = []
        self._hist_records[name].append(value)
        return np.average(self._hist_records[name][-num:])
    def prints(self, file, step):
        print("--------------------------  training monitor  --------------------------------------")

        i = 0

        test_accuracies = []

        for key in self._hist_records:
            i = i + 1
            print(key, self._hist_records[key][-1], "ave:", np.average(self._hist_records[key][-20:]))

            if i in [8, 12, 16, 20, 24, 28, 32, 36, 40, 44]:
                test_accuracies.append(np.average(self._hist_records[key][-20:]))

        file.write(str(step) + "," + str(test_accuracies[0]) + "," + str(test_accuracies[1]) + "," + str(test_accuracies[2]) + "," + str(test_accuracies[3]) + "," + str(test_accuracies[4]) + "," + str(test_accuracies[5]) + "," + str(test_accuracies[6]) + "," + str(test_accuracies[7]) + "," + str(test_accuracies[8]) + "," + str(test_accuracies[9]) + "\n")

        print("==========================  *************** ========================================")




# def save_images(imgs, size, path):
#     '''
#     Args:
#         imgs: [batch_size, image_height, image_width]
#         size: a list with tow int elements, [image_height, image_width]
#         path: the path to save images
#     '''
#     imgs = (imgs + 1.) / 2  # inverse_transform
#     return(scipy.misc.imsave(path, mergeImgs(imgs, size)))
#
#
# def mergeImgs(images, size):
#     h, w = images.shape[1], images.shape[2]
#     imgs = np.zeros((h * size[0], w * size[1], 3))
#     for idx, image in enumerate(images):
#         i = idx % size[1]
#         j = idx // size[1]
#         imgs[j * h:j * h + h, i * w:i * w + w, :] = image
#
#     return imgs


# if __name__ == '__main__':
#     X, Y = load_mnist(cfg.dataset, cfg.is_training)
#     print(X.get_shape())
#     print(X.dtype)
