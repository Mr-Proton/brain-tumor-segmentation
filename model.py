import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from functions import build_unet
from functions import load_dataset
from functions import read_x
from functions import read_y
from functions import dice_loss
from functions import dice_coef

model_path = os.path.join("/content/files", "model.h5")
csv_path = os.path.join("/content/files", "log.csv")

H=512
W=512

batch_size = 2
lr = 1e-4
num_epochs = 50

dataset_path = "/content/"
(x_train, y_train), (x_validation, y_valiadtion), (x_test, y_test) = load_dataset(dataset_path)

print(f"Train: {len(x_train)} - {len(y_train)}")
print(f"Valid: {len(x_validation)} - {len(y_valiadtion)}")
print(f"Test : {len(x_test)} - {len(y_test)}")

X_train = read_x(x_train)
Y_train = read_y(y_train)

model = build_unet((H, W, 3))
model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef])

callbacks = [
    ModelCheckpoint(model_path, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path),
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
]

model.fit(
    X_train, Y_train,
    batch_size=4,
    epochs=30,
    callbacks=callbacks
)


model.save("/content/model.h5")