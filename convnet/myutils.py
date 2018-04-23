import os,time
#import scipy
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

            if i in [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]:
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
