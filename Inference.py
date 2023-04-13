import argparse
import numpy as np
import tensorflow as tf
from PIL import Image


def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # assume input shape of (224, 224, 3)
    img_array = np.array(img) / 255.0  # normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array


def predict_class(model, img_array, threshold=0.5):
    result = model.predict(img_array)
    if result[0][0] > threshold:
        return "Saffron", result[0][0]
    else:
        return "Non-Saffron", 1 - result[0][0]


def main(args):
    # Define the path to the pre-trained model
    model_path = args.trained_model

    # Load the pre-trained model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the image
    img_array = load_image(args.load_image)

    # Make predictions on the input image
    predicted_class, probability = predict_class(model, img_array)

    # Print the predicted class label and probabilities
    print(f"Predicted Class: {predicted_class}")
    print(f"Probability of Saffron: {probability:.2%}")
    print(f"Probability of Non-Saffron: {(1-probability):.2%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_image", help="path to the image to be classified")
    ap.add_argument("--trained_model", help="path to the pre-trained model")
    args = ap.parse_args()

    main(args)