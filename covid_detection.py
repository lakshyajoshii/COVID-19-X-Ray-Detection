import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import cv2
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix


# Dataset Paths
BASE_PATH = "C:/Users/joshi/Desktop/PROJECT/Dataset"
TRAIN_PATH = os.path.join(BASE_PATH, "Train")
VAL_PATH = os.path.join(BASE_PATH, "Val")   
PRED_PATH = os.path.join(BASE_PATH, "Prediction")

# Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Image Data Generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True)

]

# Model Training
history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=2,
    callbacks=callbacks
)

# Save the final model
model.save("Detection_Covid_19.h5")

# Reload the model
model = load_model("Detection_Covid_19.h5")

# Predict and evaluate on validation data
y_actual = []
y_pred = []

# Normal images (label = 1)
for img_name in os.listdir(os.path.join(VAL_PATH, "Normal")):
    img_path = os.path.join(VAL_PATH, "Normal", img_name)
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    pred = (model.predict(img) > 0.5).astype("int32")
    y_pred.append(pred[0][0])
    y_actual.append(1)

# Covid images (label = 0)
for img_name in os.listdir(os.path.join(VAL_PATH, "Covid")):
    img_path = os.path.join(VAL_PATH, "Covid", img_name)
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    pred = (model.predict(img) > 0.5).astype("int32")
    y_pred.append(pred[0][0])
    y_actual.append(0)

# Confusion Matrix
cm = confusion_matrix(y_actual, y_pred)
class_names = ["Covid-19", "Normal"]

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap='plasma'):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.figure()
plot_confusion_matrix(cm, class_names)
plt.show()

# Plot Training Accuracy and Loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Prediction Example 1 - Covid image
covid_img_path = "C:/Users/joshi/Desktop/PROJECT/Dataset/Prediction/Covid/person94_bacteria_456.jpeg"
img = image.load_img(covid_img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
result = (model.predict(img_array) > 0.5).astype("int32")

display_img = cv2.imread(covid_img_path)
display_img = cv2.resize(display_img, (400, 400))
plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
print("This X-ray Image is of a Covid-19 patient")
print("Prediction:", "Positive For Covid-19" if result[0][0] == 0 else "Negative for Covid-19")

# Prediction Example 2 - Normal image
normal_img_path = "C:/Users/joshi/Desktop/PROJECT/Dataset/Prediction/Normal/NORMAL2-IM-0028-0001.jpeg"
img = image.load_img(normal_img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
result = (model.predict(img_array) > 0.5).astype("int32")

display_img = cv2.imread(normal_img_path)
display_img = cv2.resize(display_img, (400, 400))
plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
print("This X-ray Image is of a Normal patient")
print("Prediction:", "Negative for Covid-19" if result[0][0] == 0 else "Positive for Covid-19")
