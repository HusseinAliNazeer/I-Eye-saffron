import argparse
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout,
                           Flatten, GlobalAveragePooling2D, Input, MaxPool2D)
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Model, Sequential, load_model

# Set plotting parameters
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (15, 15)
plt.rcParams["font.size"] = 18

def parse_arguments():
    """Parses command-line arguments."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="path to the data directory")
    ap.add_argument("--train", action="store_true", help="train the model on the data")
    ap.add_argument("--ckpPath", help="path to save model checkpoints")
    ap.add_argument("--epochs", help="number of epochs to train the model", default=10, type=int)
    ap.add_argument("--load", help="load the previously trained model")
    return ap.parse_args()

def load_data(train_dir, test_dir, val_dir):
    """Loads data from directories."""
    train_gen = ImageDataGenerator(
        shear_range=0.2,
        rescale=1. / 255,
        zoom_range=[0.5, 2.0],
        rotation_range=45,
        horizontal_flip=True,
        vertical_flip=True,
        featurewise_center=True,
        samplewise_center=True,
    )

    test_gen = ImageDataGenerator(rescale=1. / 255)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        class_mode="binary",
        batch_size=32,
        shuffle=True,
    )

    test_data = test_gen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        class_mode="binary",
        batch_size=32,
        shuffle=True,
    )

    val_data = test_gen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        class_mode="binary",
        batch_size=32,
        shuffle=True,
    )

    return train_data, test_data, val_data

def create_model(load_path=None):
    """Creates a model."""
    if load_path:
        # Load a previously trained model
        model = tf.keras.models.load_model(load_path)
    else:
        # Define a new model
        base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False)
        for layer in base_model.layers:
            layer.trainable = False

        last_layer = base_model.layers[-3].output
        pool = GlobalAveragePooling2D()(last_layer)
        dense_1 = Dense(264, activation="relu")(pool)
        drop_1 = Dropout(0.4)(dense_1)
        dense_2 = Dense(128, activation="relu")(drop_1)
        drop_2 = Dropout(0.4)(dense_2)
        output = Dense(1, activation="sigmoid")(drop_2)

        model = Model(inputs=base_model.input, outputs=output)

        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    return model

def train_model(model, train_data, val_data, ckp_path, epochs):
    """Trains the model."""
    ckp = ModelCheckpoint(
        ckp_path + datetime.date.today() + "best_model.h5",
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="max",
    )

    es = EarlyStopping(
        monitor="val_accuracy", min_delta=0, patience=5, verbose=1, mode="max"
    )

    history = model.fit(
        train_data,
        epochs=epochs,
        batch_size=32,
        validation_data=val_data,
        callbacks=[ckp, es],
    )

    return history

def evaluate_model(model, test_data):
    """Evaluates the model on the test data."""
    loss, acc = model.evaluate(test_data, verbose=1)
    print("Trained model, accuracy on test data: {:5.2f}%".format(100 * acc))

    y_test = np.concatenate([test_data.next()[1] for i in range(test_data.len())])
    y_pred = model.predict_generator(test_data)
    print(classification_report(y_test, y_pred))

def main():
    args = parse_arguments()

    train_dir = os.path.join(args.path, "train")
    test_dir = os.path.join(args.path, "test")
    val_dir = os.path.join(args.path, "val")

    train_data, test_data, val_data = load_data(train_dir, test_dir, val_dir)

    if args.load:
        model = create_model(args.load)
    else:
        model = create_model()

    if args.train:
        ckp_path = args.ckpPath or ""
        history = train_model(model, train_data, val_data, ckp_path, args.epochs)

    evaluate_model(model, test_data)

if __name__ == "__main__":
    main()