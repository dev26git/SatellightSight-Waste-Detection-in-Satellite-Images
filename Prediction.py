import cv2
import pandas as pd
import numpy as np
import pickle
from resources import Utilities

# Load mean_std_df
mean_std_df = pd.read_csv('resources/mean_std_df.csv')

# Load model
loaded_model = pickle.load(open('resources/models/RandomForestClassifier.sav', 'rb'))


# Defining a custom standard scaler to scale features
def myStandardScaler(features):
    scaled_features = []
    for i, value in enumerate(features):
        mean = mean_std_df.iat[i, 1]
        std = mean_std_df.iat[i, 2]
        scaled_features.append((value - mean) / std)
    return np.array(scaled_features)


def getAnnotatedImage(img_rgb):
    positive_windows = []
    for window_coord in Utilities.get_window_coords(img_rgb, 150, 150):

        # get cropped window from img
        # cropped_img = img[top:bottom, left:right]
        window = img_rgb[window_coord[0]:window_coord[1], window_coord[2]:window_coord[3]]

        # get features of window
        features = Utilities.get_features(window)
        # Feature Scaling
        features = myStandardScaler(features)

        # Predict class using model
        predicted_class = loaded_model.predict(np.reshape(features, (1, -1)))
        # print(predicted_class)

        if predicted_class[0] == 1:
            # append coordinates into positive_windows list if waste detected
            positive_windows.append(window_coord)

    for window_coord in positive_windows:
        # Overlay the bounding box on the image
        cv2.rectangle(img_rgb, (window_coord[2], window_coord[0]), (window_coord[3], window_coord[1]), (0, 255, 0), 2)

    return img_rgb
