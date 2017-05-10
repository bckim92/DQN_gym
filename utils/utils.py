from scipy.misc import imresize
import numpy as np
import tensorflow as tf


def atari_preprocessing(raw_image, width, height):
    gray_image = np.dot(raw_image[..., :3], [0.299, 0.587, 0.114]) / 255
    resized_image = imresize(gray_image, [width, height])
    return resized_image


def get_histo(values, bins=1000):
    """Generate tf.HistogramProto() from list of values"""
    if isinstance(values, list):
        values = np.array(values)

    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    return hist
