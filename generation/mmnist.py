import sys

sys.path.append("../")
from input_utils import load_mnist, load_mmnist, load_mmnist4, load_submmnist
import numpy as np
from random import randint
import os
from matplotlib import pyplot as plt
from scipy import ndimage


def _mergeMNIST(img1, img2, shift_pix, rand_shift, rot_range, corot):
    assert shift_pix <= 8, "can only shift up to 8 with 32x32 MMNIST and 28x28 original MNIST"
    if corot:
        rot_deg = randint(rot_range[0], rot_range[1])
        img1 = ndimage.rotate(img1, rot_deg, reshape=False)
        img2 = ndimage.rotate(img2, rot_deg, reshape=False)
    else:
        rot_deg = randint(rot_range[0], rot_range[1])
        img1 = ndimage.rotate(img1, rot_deg, reshape=False)
        rot_deg = randint(rot_range[0], rot_range[1])
        img2 = ndimage.rotate(img2, rot_deg, reshape=False)

    if rand_shift:
        shift_pix = randint(0, shift_pix)
    # merge two images from MNIST into one
    canvas = np.zeros([36, 36])
    canvas[:28, :28] += img1
    canvas[shift_pix:shift_pix + 28, shift_pix:shift_pix + 28] += img2
    canvas = np.clip(canvas, 0, 1)
    return canvas


def _merge_many_MNIST(images, shift_pixels, rotation_range, use_shifting=True, rotate=True):
    """
    Merges an arbitrary number of MNIST images.

    :param images: a list of MNIST images
    :param shift_pixels: the (maximum) number of pixels to shift each image by
    :param rotation_range: a tuple containing the minimum and maximum rotation angle, in degrees, for each MNIST image
    :param use_shifting: whether to shift the images
    :param rotate: whether to rotate
    :return:
    """
    assert shift_pixels <= 8, "Each MNIST image can be shifted a maximum of 8 pixels, since the input images are 28x28 and the output images are 36x36. You are requesting a shift of " + (shift_pixels) + " pixels."

    if rotate:
        for i in range(0, images.__len__()):
            rot_deg = randint(rotation_range[0], rotation_range[1])
            images[i] = ndimage.rotate(images[i], rot_deg, reshape=False)

    if use_shifting:
        canvas = np.zeros([36, 36])

        for i in range(0, images.__len__()):
            shift_pixels = randint(0, shift_pixels)
            print(shift_pixels)
            canvas[shift_pixels:shift_pixels + 28, shift_pixels:shift_pixels + 28] += images[i]
            canvas = np.clip(canvas, 0, 1)
    else:
        canvas = np.zeros([36, 36])

        for i in range(0, images.__len__()):
            canvas[:28, :28] += images[i]
            canvas = np.clip(canvas, 0, 1)

    return canvas



def _MNIST2MMNIST(X, Y, num_samples, outpath, shift_pix, rand_shift=False, rot_range=[0, 0], corot=True):
    # saves MMNIST from MNIST backend
    assert X.shape[0] == Y.shape[0], "number of images and labels should be equal"
    n_MNIST = X.shape[0]

    # samples MNIST to make MMNIST
    X_MMNIST = []
    Y_MMNIST = []
    while len(Y_MMNIST) < num_samples:
        if (len(Y_MMNIST) % 1000 == 1):
            print(str(100 * len(Y_MMNIST) / (num_samples - 1)) + "% done")

        # pick two random train images with different labels
        idx1 = len(Y_MMNIST) % (n_MNIST - 1)
        idx2 = randint(0, n_MNIST - 1)
        # print "debug:", Y[idx1], Y[idx2]
        if Y[idx1] != Y[idx2]:
            # merge two images together
            Y_MMNIST.append([Y[idx1], Y[idx2]])
            merged_img = _mergeMNIST(X[idx1, :, :, 0], X[idx2, :, :, 0], shift_pix=shift_pix, rand_shift=rand_shift,
                                     rot_range=rot_range, corot=corot)
            X_MMNIST.append(merged_img)
        # print "next img", len(Y_MMNIST)

    # save imgMMNIST and lblMMNIST
    X_MMNIST = np.asarray(X_MMNIST)
    Y_MMNIST = np.asarray(Y_MMNIST, dtype=np.int32)  # convert label to int32
    # convert image to uint8
    X_MMNIST = 255. * X_MMNIST
    X_MMNIST = X_MMNIST.astype(np.uint8)
    # save images and labels
    np.ndarray.tofile(X_MMNIST, outpath + 'X')
    np.ndarray.tofile(Y_MMNIST, outpath + 'Y')
    return None


def _mnist_to_mmnist4(X, Y, num_samples, outpath, shift_pixels, use_shifting=False, rotation_range=None, rotate=False):
    # Initial checks
    assert X.shape[0] == Y.shape[0], "The number of images (" + X.shape[0] + ") and labels (" + Y.shape[0] + ") should be equal!"

    if rotation_range is None:
        rotation_range = [0, 0]

    num_digits = X.shape[0]


    # Sample the MNIST images to create the MMNIST dataset
    X_MMNIST = []
    Y_MMNIST = []

    while len(Y_MMNIST) < num_samples:
        # Pick four random training images with different labels
        idx1 = len(Y_MMNIST) % (num_digits - 1)
        idx2 = randint(0, num_digits - 1)
        idx3 = randint(0, num_digits - 1)
        idx4 = randint(0, num_digits - 1)

        while Y[idx1] == Y[idx2] or Y[idx1] == Y[idx3] or Y[idx1] == Y[idx4] or Y[idx2] == Y[idx3] or Y[idx2] == Y[idx4] or Y[idx3] == Y[idx4]:
            idx1 = len(Y_MMNIST) % (num_digits - 1)
            idx2 = randint(0, num_digits - 1)
            idx3 = randint(0, num_digits - 1)
            idx4 = randint(0, num_digits - 1)

        print "debug:", Y[idx1], Y[idx2], Y[idx3], Y[idx4]

        assert Y[idx1] != Y[idx2] != Y[idx3] != Y[idx4]

        # merge four images together
        Y_MMNIST.append([Y[idx1], Y[idx2], Y[idx3], Y[idx4]])
        images = [
            X[idx1, :, :, 0],
            X[idx2, :, :, 0],
            X[idx3, :, :, 0],
            X[idx4, :, :, 0]
        ]
        merged_img = _merge_many_MNIST(images, shift_pixels, rotation_range, use_shifting=use_shifting, rotate=rotate)
        X_MMNIST.append(merged_img)
        print "next img", len(Y_MMNIST)

    # save imgMMNIST and lblMMNIST
    X_MMNIST = np.asarray(X_MMNIST)
    Y_MMNIST = np.asarray(Y_MMNIST, dtype=np.int32)  # convert label to int32
    # convert image to uint8
    X_MMNIST = 255. * X_MMNIST
    X_MMNIST = X_MMNIST.astype(np.uint8)
    # save images and labels
    np.ndarray.tofile(X_MMNIST, outpath + 'X')
    np.ndarray.tofile(Y_MMNIST, outpath + 'Y')
    return None


def _mnist_to_mmnist2(X, Y, num_samples, outpath, shift_pixels, exclude_digit, use_shifting=False, rotation_range=None, rotate=False):
    """

    :param X:
    :param Y:
    :param num_samples:
    :param outpath:
    :param shift_pixels:
    :param exclude_digit: A single digit to not include in the dataset.
    :param use_shifting:
    :param rotation_range:
    :param rotate:
    :return:
    """
    # Initial checks
    assert X.shape[0] == Y.shape[0], "The number of images (" + X.shape[0] + ") and labels (" + Y.shape[0] + ") should be equal!"

    if rotation_range is None:
        rotation_range = [0, 0]

    num_handwritten_digits = X.shape[0]


    # Sample the MNIST images to create the MMNIST dataset
    X_MMNIST = []
    Y_MMNIST = []

    while len(Y_MMNIST) < num_samples:
        # Pick four random training images with different labels
        idx1 = len(Y_MMNIST) % (num_handwritten_digits - 1)
        idx2 = randint(0, num_handwritten_digits - 1)

        while Y[idx1] == exclude_digit or Y[idx2] == exclude_digit or Y[idx1] == Y[idx2]:
            idx1 = randint(0, num_handwritten_digits - 1)
            idx2 = randint(0, num_handwritten_digits - 1)

        print "debug:", Y[idx1], Y[idx2]

        # merge four images together
        Y_MMNIST.append([Y[idx1], Y[idx2]])
        images = [
            X[idx1, :, :, 0],
            X[idx2, :, :, 0]
        ]
        merged_img = _merge_many_MNIST(images, shift_pixels, rotation_range, use_shifting=use_shifting, rotate=rotate)
        # merged_img = _mergeMNIST(X[idx1, :, :, 0], X[idx2, :, :, 0], shift_pix=shift_pixels, rand_shift=use_shifting, rot_range=rotation_range, corot=True)
        X_MMNIST.append(merged_img)
        print "next img", len(Y_MMNIST)

    # save imgMMNIST and lblMMNIST
    X_MMNIST = np.asarray(X_MMNIST)
    Y_MMNIST = np.asarray(Y_MMNIST, dtype=np.int32)  # convert label to int32
    # convert image to uint8
    X_MMNIST = 255. * X_MMNIST
    X_MMNIST = X_MMNIST.astype(np.uint8)
    # save images and labels
    np.ndarray.tofile(X_MMNIST, outpath + 'X')
    np.ndarray.tofile(Y_MMNIST, outpath + 'Y')
    return None


def choose_ld_examples(Y, low_data_digit, num_examples):
    examples = []

    num_handwritten_digits = Y.shape[0]

    while examples.__len__() < num_examples:
        idx = randint(0, num_handwritten_digits - 1)

        if Y[idx] != low_data_digit or idx in examples:
            continue

        examples.append(idx)

    return examples


def _mnist_to_mmnist2_low_data(X, Y, num_samples, outpath, shift_pixels, low_data_digit, num_examples=100, use_shifting=False, rotation_range=None, rotate=False):
    """

    :param X:
    :param Y:
    :param num_samples:
    :param outpath:
    :param shift_pixels:
    :param exclude_digit: A single digit to not include in the dataset.
    :param use_shifting:
    :param rotation_range:
    :param rotate:
    :return:
    """
    # Initial checks
    assert X.shape[0] == Y.shape[0], "The number of images (" + X.shape[0] + ") and labels (" + Y.shape[0] + ") should be equal!"

    if rotation_range is None:
        rotation_range = [0, 0]

    num_handwritten_digits = X.shape[0]


    # Sample the MNIST images to create the MMNIST dataset
    X_MMNIST = []
    Y_MMNIST = []

    # Pick 100 examples of our low data digit
    low_data_examples = choose_ld_examples(Y, low_data_digit, num_examples)

    while len(Y_MMNIST) < num_samples:
        # Pick four random training images with different labels
        idx1 = len(Y_MMNIST) % (num_handwritten_digits - 1)
        idx2 = randint(0, num_handwritten_digits - 1)


        # If one of the two examples we pick is not in the low-data set, choose one from the low_data set instead
        # (don't just continue since we want about an equal distribution of training digits)

        if Y[idx1] == low_data_digit and idx1 not in low_data_examples:
            idx1 = low_data_examples[randint(0, low_data_examples.__len__() - 1)]

        if Y[idx2] == low_data_digit and idx2 not in low_data_examples:
            idx2 = low_data_examples[randint(0, low_data_examples.__len__() - 1)]

        print "debug:", Y[idx1], Y[idx2]

        # merge four images together
        Y_MMNIST.append([Y[idx1], Y[idx2]])
        images = [
            X[idx1, :, :, 0],
            X[idx2, :, :, 0]
        ]
        merged_img = _merge_many_MNIST(images, shift_pixels, rotation_range, use_shifting=use_shifting, rotate=rotate)
        # merged_img = _mergeMNIST(X[idx1, :, :, 0], X[idx2, :, :, 0], shift_pix=shift_pixels, rand_shift=use_shifting, rot_range=rotation_range, corot=True)
        X_MMNIST.append(merged_img)
        print "next img", len(Y_MMNIST)

    # save imgMMNIST and lblMMNIST
    X_MMNIST = np.asarray(X_MMNIST)
    Y_MMNIST = np.asarray(Y_MMNIST, dtype=np.int32)  # convert label to int32
    # convert image to uint8
    X_MMNIST = 255. * X_MMNIST
    X_MMNIST = X_MMNIST.astype(np.uint8)
    # save images and labels
    np.ndarray.tofile(X_MMNIST, outpath + 'X')
    np.ndarray.tofile(Y_MMNIST, outpath + 'Y')
    return None



def random(min, max, ignore):
    """
    Chooses a random number between min and max, that is not in the set of numbers to ignore.

    :param min: The minimum bound
    :param max: The maximum bound
    :param ignore: Numbers to ignore. Assumed to be in range [min, max]
    :return:
    """
    assert ignore.__len__() <= max - min, "I can't choose a number since you're disallowing too many!"

    while True:
        chosen = randint(min, max)
        if chosen not in ignore:
            return chosen



def genMMNIST(mnistpath, outpath, samples_tr=200000, samples_te=10000):
    trX, trY, teX, teY = load_mnist(mnistpath)
    # generate training samples
    _MNIST2MMNIST(trX, trY, samples_tr, os.path.join(outpath, 'tr'), shift_pix=8, rand_shift=True)
    # generate test samples
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'te0'), shift_pix=0)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'te1'), shift_pix=1)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'te2'), shift_pix=2)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'te3'), shift_pix=3)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'te4'), shift_pix=4)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'te5'), shift_pix=5)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'te6'), shift_pix=6)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'te7'), shift_pix=7)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'te8'), shift_pix=8)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'teR30'), shift_pix=8, rot_range=[0, 30])
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'teR60'), shift_pix=8, rot_range=[30, 60])
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'teR30R'), shift_pix=8, rot_range=[0, 30], corot=False)
    _MNIST2MMNIST(teX, teY, samples_te, os.path.join(outpath, 'teR60R'), shift_pix=8, rot_range=[30, 60], corot=False)


def genMMNIST_with_full_test(mnistpath, outpath, exclude_digit_for_test, samples_tr=200000, samples_te=10000):
    trX, trY, teX, teY = load_mnist(mnistpath)

    # Generate the training examples (full MMNIST!)
    _MNIST2MMNIST(trX, trY, samples_tr, os.path.join(outpath, 'tr'), shift_pix=8, rand_shift=True)

    # First, generate the nine-class test sets (with the excluded digit missing)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes0'), exclude_digit=exclude_digit_for_test, shift_pixels=0,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes1'), exclude_digit=exclude_digit_for_test, shift_pixels=1,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes2'), exclude_digit=exclude_digit_for_test, shift_pixels=2,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes3'), exclude_digit=exclude_digit_for_test, shift_pixels=3,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes4'), exclude_digit=exclude_digit_for_test, shift_pixels=4,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes5'), exclude_digit=exclude_digit_for_test, shift_pixels=5,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes6'), exclude_digit=exclude_digit_for_test, shift_pixels=6,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes7'), exclude_digit=exclude_digit_for_test, shift_pixels=7,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes8'), exclude_digit=exclude_digit_for_test, shift_pixels=8,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tesR30R'), exclude_digit=exclude_digit_for_test,
                      shift_pixels=8, rotation_range=[0, 30], use_shifting=True, rotate=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tesR60R'), exclude_digit=exclude_digit_for_test,
                      shift_pixels=8, rotation_range=[30, 60], use_shifting=True, rotate=True)

    # Then, generate the ten-class test sets (containing all digits)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef0'), exclude_digit=None, shift_pixels=0,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef1'), exclude_digit=None, shift_pixels=1,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef2'), exclude_digit=None, shift_pixels=2,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef3'), exclude_digit=None, shift_pixels=3,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef4'), exclude_digit=None, shift_pixels=4,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef5'), exclude_digit=None, shift_pixels=5,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef6'), exclude_digit=None, shift_pixels=6,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef7'), exclude_digit=None, shift_pixels=7,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef8'), exclude_digit=None, shift_pixels=8,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tefR30R'), exclude_digit=None, shift_pixels=8,
                      rotation_range=[0, 30], use_shifting=True, rotate=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tefR60R'), exclude_digit=None, shift_pixels=8,
                      rotation_range=[30, 60], use_shifting=True, rotate=True)


def gensubmmnist(mnistpath, outpath, exclude_digit, samples_tr=200000, samples_te=10000):
    trX, trY, teX, teY = load_mnist(mnistpath)
    # generate training samples
    _mnist_to_mmnist2(trX, trY, samples_tr, os.path.join(outpath, 'tr'), exclude_digit=exclude_digit, shift_pixels=8, use_shifting=True)

    # First, generate the nine-class test sets (with the excluded digit missing)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes0'), exclude_digit=exclude_digit, shift_pixels=0, use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes1'), exclude_digit=exclude_digit, shift_pixels=1, use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes2'), exclude_digit=exclude_digit, shift_pixels=2, use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes3'), exclude_digit=exclude_digit, shift_pixels=3, use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes4'), exclude_digit=exclude_digit, shift_pixels=4, use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes5'), exclude_digit=exclude_digit, shift_pixels=5, use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes6'), exclude_digit=exclude_digit, shift_pixels=6, use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes7'), exclude_digit=exclude_digit, shift_pixels=7, use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes8'), exclude_digit=exclude_digit, shift_pixels=8, use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tesR30R'), exclude_digit=exclude_digit, shift_pixels=8, rotation_range=[0, 30], use_shifting=True, rotate=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tesR60R'), exclude_digit=exclude_digit, shift_pixels=8, rotation_range=[30, 60], use_shifting=True, rotate=True)

    # Then, generate the ten-class test sets (containing all digits)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef0'), exclude_digit=None, shift_pixels=0, use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef1'), exclude_digit=None, shift_pixels=1, use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef2'), exclude_digit=None, shift_pixels=2, use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef3'), exclude_digit=None, shift_pixels=3, use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef4'), exclude_digit=None, shift_pixels=4, use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef5'), exclude_digit=None, shift_pixels=5, use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef6'), exclude_digit=None, shift_pixels=6, use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef7'), exclude_digit=None, shift_pixels=7, use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef8'), exclude_digit=None, shift_pixels=8, use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tefR30R'), exclude_digit=None, shift_pixels=8, rotation_range=[0, 30], use_shifting=True, rotate=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tefR60R'), exclude_digit=None, shift_pixels=8, rotation_range=[30, 60], use_shifting=True, rotate=True)


def genldmmnist(mnistpath, outpath, ld_digit, num_examples, samples_tr=200000, samples_te=10000):
    trX, trY, teX, teY = load_mnist(mnistpath)
    # generate training samples
    _mnist_to_mmnist2_low_data(trX, trY, samples_tr, os.path.join(outpath, 'tr'), low_data_digit=ld_digit, num_examples=num_examples, shift_pixels=8, use_shifting=True)

    # First, generate the nine-class test sets (with the excluded digit missing)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes0'), exclude_digit=ld_digit, shift_pixels=0,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes1'), exclude_digit=ld_digit, shift_pixels=1,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes2'), exclude_digit=ld_digit, shift_pixels=2,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes3'), exclude_digit=ld_digit, shift_pixels=3,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes4'), exclude_digit=ld_digit, shift_pixels=4,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes5'), exclude_digit=ld_digit, shift_pixels=5,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes6'), exclude_digit=ld_digit, shift_pixels=6,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes7'), exclude_digit=ld_digit, shift_pixels=7,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tes8'), exclude_digit=ld_digit, shift_pixels=8,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tesR30R'), exclude_digit=ld_digit,
                      shift_pixels=8, rotation_range=[0, 30], use_shifting=True, rotate=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tesR60R'), exclude_digit=ld_digit,
                      shift_pixels=8, rotation_range=[30, 60], use_shifting=True, rotate=True)

    # Then, generate the ten-class test sets (containing all digits)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef0'), exclude_digit=None, shift_pixels=0,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef1'), exclude_digit=None, shift_pixels=1,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef2'), exclude_digit=None, shift_pixels=2,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef3'), exclude_digit=None, shift_pixels=3,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef4'), exclude_digit=None, shift_pixels=4,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef5'), exclude_digit=None, shift_pixels=5,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef6'), exclude_digit=None, shift_pixels=6,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef7'), exclude_digit=None, shift_pixels=7,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tef8'), exclude_digit=None, shift_pixels=8,
                      use_shifting=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tefR30R'), exclude_digit=None, shift_pixels=8,
                      rotation_range=[0, 30], use_shifting=True, rotate=True)
    _mnist_to_mmnist2(teX, teY, samples_te, os.path.join(outpath, 'tefR60R'), exclude_digit=None, shift_pixels=8,
                      rotation_range=[30, 60], use_shifting=True, rotate=True)


def genMMNIST4(mnistpath, outpath, samples_tr=200000, samples_te=10000):
    trX, trY, teX, teY = load_mnist(mnistpath)
    # generate training samples
    _mnist_to_mmnist4(trX, trY, samples_tr, os.path.join(outpath, 'tr'), shift_pixels=8, use_shifting=True)
    # generate test samples
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'te0'), shift_pixels=0, use_shifting=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'te1'), shift_pixels=1, use_shifting=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'te2'), shift_pixels=2, use_shifting=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'te3'), shift_pixels=3, use_shifting=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'te4'), shift_pixels=4, use_shifting=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'te5'), shift_pixels=5, use_shifting=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'te6'), shift_pixels=6, use_shifting=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'te7'), shift_pixels=7, use_shifting=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'te8'), shift_pixels=8, use_shifting=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'teR30R'), shift_pixels=8, rotation_range=[0, 30], use_shifting=True, rotate=True)
    _mnist_to_mmnist4(teX, teY, samples_te, os.path.join(outpath, 'teR60R'), shift_pixels=8, rotation_range=[30, 60], use_shifting=True, rotate=True)


dataset_path = '/home/andrew/datasets/ldtest'
# genMMNIST_with_full_test('/home/andrew/mnist', dataset_path, exclude_digit_for_test=6, samples_tr=200000, samples_te=10000)
# mmnist = load_mmnist(dataset_path, samples_tr=200000, samples_te=10000)
# gensubmmnist('/home/andrew/mnist', dataset_path, exclude_digit=6, samples_tr=200000, samples_te=10000)
# mmnist = load_mmnist(dataset_path, samples_tr=200000, samples_te=10000)
genldmmnist('/home/andrew/mnist', dataset_path, ld_digit=6, num_examples=1, samples_tr=2000, samples_te=100)
mmnist = load_mmnist(dataset_path, samples_tr=2000, samples_te=100)

for i in range(0, 100):
    image = mmnist["trX"][i][:, :, 0]
    digits = mmnist["trY"][i]
    if 6 in digits:
        print "digits:", digits
        plt.imshow(image)
        plt.savefig('example' + str(i) + '.png')
# print "showitng digits:", mmnist[1][11,:]
# plt.show()
# plt.imshow(mmnist[2][11,:,:,0])
# print "showitng digits:", mmnist[3][11,:]
# plt.show()
