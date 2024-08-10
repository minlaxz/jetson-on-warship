import warnings

warnings.filterwarnings("ignore")
import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

np.random.seed(2)

label_number = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "A": 10,
    "B": 11,
    "C": 12,
    "D": 13,
    "E": 14,
    "F": 15,
    "G": 16,
    "H": 17,
    "I": 18,
    "J": 19,
    "K": 20,
    "L": 21,
    "M": 22,
    "N": 23,
    "P": 24,
    "Q": 25,
    "R": 26,
    "S": 27,
    "T": 28,
    "U": 29,
    "V": 30,
    "W": 31,
    "X": 32,
    "Y": 33,
    "Z": 34,
}
label_word = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]
dataframed = pd.DataFrame(columns=["path", "label"])
for x in range(0, len(label_word)):
    temp = "./letters-dataset/" + label_word[x] + "/"
    for dirname, _, filenames in os.walk(temp):
        for filename in filenames:
            name = filename
            label = label_number[label_word[x]]
            dataframed.loc[len(dataframed)] = [temp + "/" + name, label]

print(dataframed.head())
print("Shape of the Dataset = ", dataframed.shape)


train, test = train_test_split(dataframed, test_size=0.2, random_state=42)
test, valid = train_test_split(test, test_size=0.5, random_state=42)
# Train = 80%
# Test = 10%
# valid = 10%

print(train.head())
print(test.head())
print(valid.head())


def size_regulator(df, image_column="path", target_size=(100, 100)):
    images = []
    for path in df[image_column]:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, target_size)
        images.append(image)
    return np.array(images)


X_Train = size_regulator(train)
X_Test = size_regulator(test)
X_Valid = size_regulator(valid)

y_Train = train
y_Test = test
y_Valid = valid

y_Train = y_Train.drop(["path"], axis=1)
y_Test = y_Test.drop(["path"], axis=1)
y_Valid = y_Valid.drop(["path"], axis=1)

y_Train.head()
y_Train_encoded = to_categorical(y_Train)
y_Test.head()
y_Test_encoded = to_categorical(y_Test)
y_Valid.head()
y_Valid_encoded = to_categorical(y_Valid)

y_Train_encoded = np.asarray(y_Train_encoded).astype("float32").reshape((-1, 35))
y_Test_encoded = np.asarray(y_Test_encoded).astype("float32").reshape((-1, 35))
y_Valid_encoded = np.asarray(y_Valid_encoded).astype("float32").reshape((-1, 35))


X_Train = X_Train.reshape(-1, 100, 100, 1)
y_Train = np.array(y_Train)


X_Test = X_Test.reshape(-1, 100, 100, 1)
y_Test_np = np.array(y_Test)

X_Valid = X_Valid.reshape(-1, 100, 100, 1)
y_Valid_np = np.array(y_Valid)

print("images in Train Shape = ", X_Train.shape)
print("images in Test Shape = ", X_Test.shape)
print("images in Valid Shape = ", X_Valid.shape)

Generation_1 = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
)


Generation_1.fit(X_Train)


model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        strides=1,
        padding="same",
        activation="relu",
        input_shape=(100, 100, 1),
    )
)
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(units=35, activation="softmax"))
print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

learning_rate_callback = ReduceLROnPlateau(
    monitor="val_accuracy", patience=2, verbose=1, factor=0.3, min_lr=0.000001
)
hisotry = model.fit(
    Generation_1.flow(X_Train, y_Train_encoded, batch_size=32),
    validation_data=Generation_1.flow(X_Valid, y_Valid_encoded),
    callbacks=[learning_rate_callback],
    epochs=50,
)

print("Loss of the model is - ", model.evaluate(X_Test, y_Test_encoded)[0])
print(
    "Accuracy of the model is - ", model.evaluate(X_Test, y_Test_encoded)[1] * 100, "%"
)

listed = list(np.where(dataframed))
listed = listed[0].tolist()
listed = list(set(listed))
y1 = model.predict(X_Test)
for x in range(0, 50):
    plt.subplots()
    plt.imshow(X_Test[x], cmap="Greys_r")
    prediction_answer = np.argmax(y1[x])
    print(f"Predicted label :{label_word[prediction_answer]}")
    print(f"True label :{label_word[y_Test.iloc[int(listed[x])]['label']]}")
