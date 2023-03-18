# The Saffrosystems ‘i-Eye®’
This is a Python script for building a Convolutional Neural Network (CNN) for Fake Saffron Detection. The model uses transfer learning with the MobileNetV2 architecture to classify images into two categories. The script loads a dataset containing images in separate directories for training, testing, and validation.

## Usage
To use this script, you need to have Python 3 and the following libraries installed:

* `tensorflow`
* `numpy`
* `pandas`
* `seaborn`
* `matplotlib`
The script can be executed from the command line with the following arguments:

--path: the path to the directory containing the training, testing, and validation data directories
--train: to train the model on the dataset
--load: to load a previously trained model for fine-tuning or prediction
## Arguments
`--path`
This argument is required and specifies the path to the directory containing the training, testing, and validation data directories.

`--train`
This argument is optional and specifies whether to train the model on the dataset. If specified, the script will train the model on the data and save the trained model to the specified path. If not specified, the script will assume that a trained model already exists and will load it from the specified path.

`--load`
This argument is optional and specifies whether to load a previously trained model for fine-tuning or prediction. If specified, the script will load the model from the specified path. If not specified, the script will assume that a trained model is not being loaded.

## Data Preparation
The data directory should be split into three directories: train, test, and val. Each directory should contain subdirectories for each class of images.

## Model Architecture
The model uses transfer learning with the MobileNetV2 architecture. The base model layers are frozen, and a top classifier is added to train only the classifier. The top classifier consists of three dense layers with ReLU activation and dropout, followed by an output layer with sigmoid activation. The model is compiled with the Adam optimizer and binary cross-entropy loss.

## Callbacks
The script defines two callbacks for the model:

* `ModelCheckpoint`: to save the best model during training
* `EarlyStopping` : to stop training if the validation accuracy does not improve for a specified number of epochs
## Evaluation
After training the model, the script evaluates the performance on the test data and prints the accuracy score. Additionally, the script prints a classification report for the model's performance on the test data.

## Credits
This script was developed by [Saffron Systems](https://saffrosystems.com/) as part of a computer vision project.
