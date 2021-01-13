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

class CreateModel(tf.keras.Model):
    def __init__(self, label_dims):
        super(CreateModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(units=256, use_bias=True, kernel_initializer='he_uniform')
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.drop1 = tf.keras.layers.Dropout(rate=0.7)

        self.dense2 = tf.keras.layers.Dense(units=512, use_bias=True, kernel_initializer='he_uniform')
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.drop2 = tf.keras.layers.Dropout(rate=0.7)

        self.dense3 = tf.keras.layers.Dense(units=label_dims, use_bias=True, kernel_initializer='he_uniform')
        self.batch_norm3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        net = self.flatten(inputs)

        net = self.dense1(net)
        net = self.relu1(net)
        net = self.batch_norm1(net)

        net = self.dense2(net)
        net = self.relu2(net)
        net = self.batch_norm2(net)

        net = self.dense3(net)
        net = self.batch_norm3(net)

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

learning_rate = 0.001
batch_size = 128
train_dataset_num = len(train_x) // batch_size

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).\
    shuffle(buffer_size=100000).\
    prefetch(buffer_size=batch_size).\
    batch(batch_size, drop_remainder=True)

test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).\
    shuffle(buffer_size=100000).\
    prefetch(buffer_size=len(test_x)).\
    batch(len(test_x), drop_remainder=True)

label_dim = 10

model = CreateModel(label_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
checkpoint_prefix = os.path.join(checkpoint_dir, 'mnist_checkpoint')
checkpoint = tf.train.Checkpoint(dnn=model)

epoch = 1
for i in range(epoch):
    for idx, (images, labels) in enumerate(train_dataset):
        grads = grad(model, images, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        
        loss = loss_fn(model, images, labels)
        accuracy = accuracy_fn(model, images, labels)
        
        print('Epoch: {:5}, Process: {:5}/{}, Loss: {:5.5f}, Accuracy: {:5.1f}%'.format(i, idx, train_dataset_num, loss, accuracy))

checkpoint.save(file_prefix=checkpoint_prefix)

checkpoint.restore(os.path.join(checkpoint_dir, 'mnist_checkpoint-1'))

for images, labels in test_dataset:
    accuracy = accuracy_fn(model, images, labels)
print('Accuracy: {:.1f}%'.format(accuracy))
