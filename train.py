import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

from functions import read_x_test
from functions import read_y_test
from functions import save_results
from functions import load_dataset


model = keras.models.load_model("model.h5")

dataset_path = "/content/"
(x_train, y_train), (x_validation, y_valiadtion), (x_test, y_test) = load_dataset(dataset_path)

X_test = read_x_test(x_test)
Y_test = read_y_test(y_test)

SCORE = []
for i in range(len(X_test)):
    """ Extracting the name """
    name = x_test[i].split("/")[-1]

    """ Image manipulation """
    x = X_test[i]/255.0                         ## [H, w, 3]
    x = np.expand_dims(x, axis=0)

    """ Prediction """
    y_pred = model.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred >= 0.5
    y_pred = y_pred.astype(np.int32)

    """ Saving the prediction """
    save_image_path = os.path.join("/content/results/", name)
    save_results(X_test[i], Y_test[i], y_pred, save_image_path)

    """ Flatten the array """
    mask = Y_test[i]/255.0
    mask = (mask > 0.5).astype(np.int32).flatten()
    y_pred = y_pred.flatten()

    """ Calculating the metrics values """
    f1_value = f1_score(mask, y_pred, labels=[0, 1], average="binary")
    jac_value = jaccard_score(mask, y_pred, labels=[0, 1], average="binary")
    recall_value = recall_score(mask, y_pred, labels=[0, 1], average="binary", zero_division=0)
    precision_value = precision_score(mask, y_pred, labels=[0, 1], average="binary", zero_division=0)
    SCORE.append([name, f1_value, jac_value, recall_value, precision_value])

""" Metrics values """
score = [s[1:]for s in SCORE]
score = np.mean(score, axis=0)
print(f"F1: {score[0]:0.5f}")
print(f"Jaccard: {score[1]:0.5f}")
print(f"Recall: {score[2]:0.5f}")
print(f"Precision: {score[3]:0.5f}")

df = pd.DataFrame(SCORE, columns=["Image", "F1", "Jaccard", "Recall", "Precision"])
df.to_csv("content/files/score.csv")