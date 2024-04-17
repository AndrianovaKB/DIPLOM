import cv2
from tensorflow.keras import models, layers, regularizers
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, Activation, GlobalAveragePooling2D, Add, AveragePooling2D
from keras.optimizers import *
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_image(image):
    # Загрузка изображения
    # print(image_path)
    # image = cv2.imread(image_path)
    # Преобразование в черно-белое изображение (если изображение уже в оттенках серого, этот шаг можно пропустить)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(gray_image)
    # Изменение размеров изображения на (48, 48)
    resized_image = cv2.resize(gray_image, (48, 48))
    # Расширение измерения для создания третьей размерности (1 канал для черно-белого изображения)
    resized_image = np.expand_dims(resized_image, axis=0)
    # Нормализация значений пикселей (обычно значения пикселей приводят к диапазону [0, 1])
    normalized_image = resized_image / 255.0
    return normalized_image

# # Пример использования:
# image_path = r'C:\Users\User\Downloads\photo1713001960 (1).jpeg'
# preprocessed_image = preprocess_image(image_path)
# print(preprocessed_image)
def DCNN():
    input_shape = (48, 48, 1)
    output_class = 7

    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(output_class, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# model.load_weights(r"C:\Users\User\Documents\диплом\model_DCNN.weights.h5")
#
# print(preprocessed_image.shape)
# y_pred=model.predict(preprocessed_image)

# print(y_pred)
# emotions = ["angry", "disgut", "fear", "happy", "neutral", "sad", "surprise"]
# max_index = np.argmax(y_pred)
# max_emotion = emotions[max_index]
# print("NN answer:", max_emotion)