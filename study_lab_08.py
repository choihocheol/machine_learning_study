import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

def load_mnist():
    path = os.path.join(os.getcwd(), 'mnist.npz')
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data(path)

    # [batch_size, height, width, color_channel]
    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

    # data normalization
    train_data = train_data.astype(np.float32) / 255.0
    test_data = test_data.astype(np.float32) / 255.0

    # One hot encoding
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    return train_data, train_labels, test_data, test_labels

class DenseBNRelu(tf.keras.Model):
    def __init__(self, units):
        super(DenseBNRelu, self).__init__()
        self.dense = tf.keras.layers.Dense(units=units, kernel_initializer='he_uniform')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.drop = tf.keras.layers.Dropout(rate=0.5)

    def call(self, inputs, training=False):
        layer = self.dense(inputs)
        layer = self.batch_norm(layer)
        layer = tf.nn.relu(layer)
        layer = self.drop(layer)
        return layer

class MNISTCNN(tf.keras.Model):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=1)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=1)
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation=tf.nn.relu, kernel_initializer='he_uniform')
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=1)
        self.flat3 = tf.keras.layers.Flatten()
        self.dense4 = DenseBNRelu(units=256)
        self.dense5 = tf.keras.layers.Dense(units=10, kernel_initializer='glorot_uniform')

    def call(self, inputs, training=False):
        net = self.conv1(inputs)
        net = self.pool1(net)
        net = self.conv2(net)
        net = self.pool2(net)
        net = self.conv3(net)
        net = self.pool3(net)
        net = self.flat3(net)
        net = self.dense4(net)
        net = self.dense5(net)
        return net
    
def loss_fn(model, images, labels):
    logits = model(images, training=True)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_pred=logits, y_true=labels, from_logits=True))
    return loss

def accuracy_fn(model, images, labels):
    logits = model(images, training=False)
    prediction = tf.equal(tf.argmax(logits, -1), tf.argmax(labels, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32)) * 100
    return accuracy

def grad(model, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)
    return tape.gradient(loss, model.variables)

train_x, train_y, test_x, test_y = load_mnist()
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).\
    shuffle(buffer_size=100000).\
    prefetch(buffer_size=len(test_x)).\
    batch(len(test_x), drop_remainder=True)

label_dims = 10
model = MNISTCNN()

checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
checkpoint_prefix = os.path.join(checkpoint_dir, 'mnist_checkpoint')
checkpoint = tf.train.Checkpoint(cnn = model)

checkpoint.restore(os.path.join(checkpoint_dir, 'mnist_checkpoint-1'))

for images, labels in test_dataset:
    accuracy = accuracy_fn(model, images, labels)
print('Accuracy: {:.1f}%'.format(accuracy))
