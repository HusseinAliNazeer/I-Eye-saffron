from PIL import Image
import numpy as np
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--load_image", help = "load the previously trianed model")
ap.add_argument("--trained_model", help = "the path for the best weights")

args = ap.parse_args()


# Define the path to the pre-trained model
model_path = args.trained_model

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Define the path to the image to be classified
image_path = args.load_image

# Load the image and preprocess it for the model
img = Image.open(image_path)
img = img.resize((224, 224)) # assume input shape of (224, 224, 3)
img_array = np.array(img) / 255.0 # normalize pixel values to [0, 1]
img_array = np.expand_dims(img_array, axis=0) # add batch dimension

# Make predictions on the input image
result = model.predict(img_array)
if result[0][0] > 0.50:
  lbl_pred = "Predicted Class : " + "Saffron" 
else:
  lbl_pred = "Predicted Class : " + "Non-Saffron"
saffron_prob = "probability of Saffron : " +str(round(result[0][0]*100,2))+"%"
nsaffron_prob = "probability of Non-Saffron: " +str(round((1-result[0][0])*100,2))+"%"
# Get the predicted class labe

# Print the predicted class label
print("Predicted label:", lbl_pred)
print(saffron_prob)
print(nsaffron_prob)