from keras.models import load_model, Model
import ipywidgets as widgets
import io
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
from IPython import display
import whatimage
import pyheif
import tensorflow as tf
from keras.preprocessing import image
import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import streamlit as st
st.title("IEye-Saffron Image Classification Model")
st.header("IEye-Saffron")
st.text("Upload a brain MRI Image for image classification as saffron or non-saffron")


def load_models():
    vis_model = load_model('fun_model_1.h5')
    pred_model = load_model('vgg_model_4.h5')
    return vis_model,pred_model

def get_img_array(img, size):
    # `img` is a PIL image of size 299x299
    img = img.resize(size=size)
    # `array` is a float32 Numpy array of shape (224, 224, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 224, 224, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer 
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

def predict(image: Image.Image,alpha=0.4):

    orignal_image = img_to_array(image.resize(size=(224,224)))
    img = np.expand_dims(orignal_image, axis = 0)
    img = img / 127.5 - 1.0  
    vis_model, pred_model = load_models()  
    result = pred_model.predict(img)
    predicted_class = ""
    if result[0][0] > 0.50:
          predicted_class = "Predicted Class : " + "Saffron" 
    else:
          predicted_class = "Predicted Class : " + "Non-Saffron"
    saffron_prob = "probability of Saffron : " +str(round(result[0][0]*100,2))+"%"
    nsaffron_prob = "probability of Non-Saffron: " +str(round((1-result[0][0])*100,2))+"%"
    
    # display the uploaded image
    # Remove last layer's softmax
    vis_model.layers[-1].activation = None

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img, vis_model, 'block5_conv3')
    sup_img = save_and_display_gradcam(orignal_image,heatmap)
    return predicted_class, saffron_prob, nsaffron_prob, sup_img

uploaded_file = st.file_uploader("Choose an Image ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.',width=400, use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label,sProb,nsProb,im = predict(image)
    st.write("Pridected Class is : " + label)
    st.write(sProb)
    st.write(nsProb)
    st.image(im,caption="Visualization of what Model have learnt",width=400, use_column_width=True)