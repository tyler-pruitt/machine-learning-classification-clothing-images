#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 07:10:38 2020

@author: tylerpruitt
"""

#Import Packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Import Fashion MNIST dataset directly from TensorFlow
fashion_mnist = tf.keras.datasets.fashion_mnist

#Extract the information encoded in fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ('T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

#Preprocess the data

#Print out the first training image
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

print('max pixel value: ', train_images[0].max())
print('min pixel value: ', train_images[0].min())

#Normalize all pixels in each image [0,1]
test_images = test_images / 255

train_images = train_images / 255

#Test to see if data is stored in correct way
plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#Build the model

#Set up the layers of the model (neural network)
#1) flatten the image arrays, then assign 28*28=784 nodes
#2) connect that to another layer which ranks it on which class it best fits
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

#Compile the model
#Loss function - measures how accurate model is during training, want to minimize this
#Optimizer - how model updates based on data is dees and loss function
#Metrics - monitors training and testing steps, this one uses 
#'accuracy': fraction of images correctly classified

model.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Train the model (neural network)

#1) feed the model
model.fit(train_images, train_labels, epochs=20)

#2) model learns associations and prints data to the screen

#3) evaluate accuracy of this model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
print('\n Test loss:', test_loss)

#4) ask model to make predictions about other images in testing data
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

print('Prediction on first test image:', class_names[np.argmax(predictions[0])])
print('Correct classification for first image in test model:', class_names[test_labels[0]])

#graph this to look at the full set of 10 class predictions
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        colour = 'blue'
    else:
        colour = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
                                         color=colour)
    
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = '#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

#5) verify predictions

#look at the first image
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

#look at the twelth image
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

#plot several images with their predictions
num_rows = 5
num_columns = 3
num_images = num_rows*num_columns
plt.figure(figsize=(2*2*num_columns, 2*num_rows))

for i in range(num_images):
    plt.subplot(num_rows, 2*num_columns, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_columns, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)

plt.tight_layout()
plt.show()

#Use the trained model

#Have the model make a prediction about an image
img = test_images[1]

print(img.shape)

img = (np.expand_dims(img, 0))

print(img.shape)

#have the model predict the classification
predictions_single = probability_model.predict(img)

print(predictions_single)

#plot the prediction
plot_value_array(1, predictions_single[0], test_labels)

_ = plt.xticks(range(10), class_names, rotation=45)

#output the classification
print(class_names[np.argmax(predictions_single[0])])