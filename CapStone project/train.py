#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction import DictVectorizer
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.feature_extraction import DictVectorizer
from tensorflow.keras.applications.xception import decode_predictions
import tensorflow.lite as tflite_tf
import tflite_runtime.interpreter as tflite
import pickle
dv = DictVectorizer(sparse=False)

# data
# !kaggle datasets download -d kritikseth/fruit-and-vegetable-image-recognition
# !patoolib.extract_archive("fruit-and-vegetable-image-recognition.zip",outdir="data")

# Create a list with the filepaths for training and testing
train_dir = Path("./data/train/")
train_filepaths = list(train_dir.glob(r"**/*.jpg"))

test_dir = Path("./data/test")
test_filepaths = list(test_dir.glob(r"**/*.jpg"))

val_dir = Path("./data/validation")
val_filepaths = list(test_dir.glob(r"**/*.jpg"))

def proc_img(filepath):
    """Create a DataFrame with the filepath and the labels of the pictures"""

    labels = [str(filepath[i]).split("/")[-2] for i in range(len(filepath))]

    filepath = pd.Series(filepath, name="Filepath").astype(str)
    labels = pd.Series(labels, name="Label")

    # Concatenate filepaths and labels
    df = pd.concat([filepath, labels], axis=1)

    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1).reset_index(drop=True)

    return df

train_df = proc_img(train_filepaths)
test_df = proc_img(test_filepaths)
val_df = proc_img(val_filepaths)


# Mobilenet

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=0,
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_images = train_generator.flow_from_dataframe(
    dataframe=val_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=0,
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)


# Load the pretained model
pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
pretrained_model.trainable = False

inputs = pretrained_model.input

x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)

outputs = tf.keras.layers.Dense(36, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'model_checkpoints/fruit_vegetable.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

history = model.fit(
    train_images,
    validation_data=val_images,
    batch_size = 32,
    epochs=5,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )
    ]
)

target_size = (150, 150)
batch_size = 32
epochs = 5
learning_rate = 0.001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss = keras.losses.BinaryCrossentropy()

# save this model
model.save('fruits_vegetables_MobileNet.h5')
final_model_MN = keras.models.load_model("fruits_vegetables_MobileNet.h5")

# evaluate
final_model_MN.evaluate(test_images)

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

model.fit(train_images, epochs=epochs, validation_data=val_images)

lr_str = str(learning_rate).replace(".", "_")
model.save(f"./Final_Fruits_Vegetable_{batch_size}_epochs{epochs}_lr_{lr_str}.h5")

# save the model

converter = tf.lite.TFLiteConverter.from_keras_model(model=model)
tflite_model = converter.convert()

with open("./Final_Fruits_Vegetable.tflite", "wb") as file:
    file.write(tflite_model)
