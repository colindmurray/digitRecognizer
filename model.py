#!/usr/bin/python3
import pandas as pd
import numpy as np
import argparse
import tensorflow as tf

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

parser = argparse.ArgumentParser(description='Train a NN on MINST dataset.')
parser.add_argument('-t', '--train', dest='train', help='Path to training set')
parser.add_argument('-s', '--test', dest='test', help='Path to test set')

args = parser.parse_args()

def readCsv(path):
    return pd.read_csv(path)

def num_to_one_hot(labels, num_classes):
    """Convert label slice to one-hot matrix"""
    num_labels = labels.shape[0]
    index_offset = numpy.arrange(num_labels) * num_classes
    labels_one_hot = numpy.zeors((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot
    

"""
TODO(colin): Implement nn from scratch: https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/mnist/mnist.py
"""

"""
Pickup here (April 11): https://www.tensorflow.org/get_started/mnist/beginners
"""


if __name__ == "__main__":
    print("trainPath = " + args.train)
    print("testPath = " + args.test)
    trainSetDf = readCsv(args.train)
    testSetDf = readCsv(args.test)
     
    trainLabels = trainSetDf["labels"]




