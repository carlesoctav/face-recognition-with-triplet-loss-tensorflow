import os
from pyprojroot.here import here
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from generate_training_data import generate_triplets
from for_training_model import for_training_model, face_net_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse


def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def visualize(anchor, positive, negative):
    """Visualize a few triplets from 1 instance of the dataset.
    plot is 1x3
    """

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(anchor)
    plt.title("Anchor")
    plt.subplot(1, 3, 2)
    plt.imshow(positive)
    plt.title("Positive")
    plt.subplot(1, 3, 3)
    plt.imshow(negative)
    plt.title("Negative")
    plt.show()
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(anchor)
    axs[1].imshow(positive)
    axs[2].imshow(negative)
    plt.show()



def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )


def prepare_data(train_size = 1000):
    """
    prepare data for training as a tf.data.Dataset instance
    """
    anchor, positive, negative = generate_triplets(here("mtcnn-faces"), train_size)
    dataset_anchor = tf.data.Dataset.from_tensor_slices(anchor)
    dataset_positive = tf.data.Dataset.from_tensor_slices(positive)
    dataset_negative = tf.data.Dataset.from_tensor_slices(negative)
    dataset = tf.data.Dataset.zip((dataset_anchor, dataset_positive, dataset_negative))
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.map(preprocess_triplets)

    return dataset


def training_process(epochs, batch_size, learning_rate, margin, train_size, cache):
    """
    fine tune pre-trained model
    epochs: number of epochs
    batch_size: batch size
    learning_rate: learning rate
    margin: margin for triplet loss
    train_size: number of training images each person
    cache: cache the dataset
    """
    dataset = prepare_data()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    if cache:
        dataset = dataset.cache()
    
    model_triple_loss, embedding_model = for_training_model()
    model_triple_loss = face_net_model(model_triple_loss, margin)
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model_triple_loss.compile(optimizer=optimizer)
    model_triple_loss.fit(dataset, epochs = epochs)

    if not os.path.exists(here("fine_tune_model")):
        os.mkdir(here("fine_tune_model"))

    embedding_model.save(here("fine_tune_model/embedding_model.h5"))


def arg_parse():
    """
    argument parser
    """
    parser = argparse.ArgumentParser(description="training model arguments")
    parser.add_argument("--epochs", type = int, default = 1, help = "number of epochs")
    parser.add_argument("--batch_size", type = int, default = 32, help = "batch size")
    parser.add_argument("--learning_rate", type = float, default = 1e-5, help = "learning rate")
    parser.add_argument("--margin", type = float, default = 0.5, help = "margin for triplet loss")
    parser.add_argument("--train_size", type = int, default = 1000, help = "number of training images")
    parser.add_argument("--cache", type = bool, default = True, help = "cache the dataset")
    return parser.parse_args()

if __name__ == "__main__":
    parser = arg_parse()
    training_process(parser.epochs,
                    parser.batch_size, 
                    parser.learning_rate,
                    parser.margin, 
                    parser.train_size,
                    parser.cache)

