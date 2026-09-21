"""
Compatibility module for MNIST input_data.
Provides read_data_sets using tf.keras.datasets.mnist.
"""
import numpy as np
import collections

class DataSet:
    def __init__(self, images, labels, reshape=True):
        self._num_examples = images.shape[0]
        if reshape:
            images = images.reshape(images.shape[0], 784)
        images = images.astype(np.float32) / 255.0
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

def _dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def read_data_sets(train_dir=None, fake_data=False, one_hot=False, dtype=np.float32, reshape=True, validation_size=5000, seed=None):
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if one_hot:
        y_train = _dense_to_one_hot(y_train, 10)
        y_test = _dense_to_one_hot(y_test, 10)

    if not 0 <= validation_size <= len(x_train):
        raise ValueError(f"Validation size should be between 0 and {len(x_train)}.")

    validation_images = x_train[:validation_size]
    validation_labels = y_train[:validation_size]
    train_images = x_train[validation_size:]
    train_labels = y_train[validation_size:]

    train = DataSet(train_images, train_labels, reshape=reshape)
    validation = DataSet(validation_images, validation_labels, reshape=reshape)
    test = DataSet(x_test, y_test, reshape=reshape)

    Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
    return Datasets(train=train, validation=validation, test=test)
