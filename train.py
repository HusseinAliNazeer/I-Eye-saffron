import io
import argparse
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, MaxPool2D, Input, GlobalAveragePooling2D
from keras.models import load_model, Model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib

matplotlib.rcParams['figure.figsize'] = (15,15)
matplotlib.rcParams['font.size'] = 18

#data directory should be splitted into 3 folders train,test and val  
ap = argparse.ArgumentParser()
ap.add_argument("--path", required = True, help = "path to the data directory")
ap.add_argument("--train", help = "train the model on the data")
ap.add_argument("--load", help = "load the previously trianed model")
args = ap.parse_args()

train_path = args.path + '/train'
test_path = args.path + '/test'
validation_path = args.path + '/val'
train_gen = ImageDataGenerator(shear_range=0.2,
                               rescale = 1./255,
                               zoom_range=[0.5,2.0],
                               rotation_range=45,
                               horizontal_flip=True,
                               vertical_flip=True,
                               featurewise_center=True,
                               samplewise_center=True)
print("train path is :",train_path)
print("test path is :",test_path)
print("val path is :",validation_path)

train_data = train_gen.flow_from_directory(train_path,
                                            target_size=(224, 224),
                                            class_mode='binary',
                                            batch_size=32,
                                            shuffle=True)
test_gen = ImageDataGenerator(rescale = 1./255)

test_data = test_gen.flow_from_directory(test_path,
                                           target_size=(224, 224),
                                            class_mode='binary',
                                            batch_size=32,
                                            shuffle=True)

val_data = test_gen.flow_from_directory(validation_path,
                                           target_size=(224, 224),
                                            class_mode='binary',
                                            batch_size=32,
                                            shuffle=True)

#Define a callback for model saving and restoring
ckp = ModelCheckpoint('/content/drive/MyDrive/SaffronSystems/Model_checkpoints/func_model_2.h5',
                      monitor='val_accuracy',
                      verbose=1,
                      save_best_only=True,
                      save_weights_only=False,
                      mode='max')

es = EarlyStopping(monitor='val_accuracy',
                  min_delta=0,
                  patience=5,
                  verbose=1,
                  mode='max')
if args.load:
   # to be completed 
   # to fine tune previously trained model
   model_path = args.load 
   model = tf.keras.models.load_model(model_path)
elif args.train:
	base_model = MobileNetV2(input_shape=(224,224,3),
                          include_top=False)
	#freeze the base model layers to train only top classifier
	for layer in base_model.layers:
		layer.trainable = False
	# define the top classifier
	last_layer = base_model.layers[-3].output
	pool = GlobalAveragePooling2D()(last_layer)
	dense_1 = Dense(264,activation='relu')(pool)
	drop_1 = Dropout(0.4)(dense_1)
	dense_2 = Dense(128,activation='relu')(drop_1)
	drop_2 = Dropout(0.4)(dense_2)
	output = Dense(1,activation='sigmoid')(drop_2)
	model = Model(inputs=base_model.input,outputs=output)
	# Compile model
	model.compile(optimizer='adam', 
				loss='binary_crossentropy', 
				metrics=['accuracy'])
	#train the model
	history =  model.fit(train_data,
						epochs=5,
						batch_size=32,
						validation_data=val_data,
						callbacks=[ckp,es])
	#evaluate model performance on test data
	loss, acc = model.evaluate(test_data, verbose=1)
	print("trained model, accuracy: {:5.2f}%".format(100 * acc))
	## measuring model performance 
	y_test=np.concatenate([test_data.next()[1] for i in range(test_data.__len__())])
	y_pred = model.predict_generator(test_data)
	print(classification_report(y_test, y_pred))


