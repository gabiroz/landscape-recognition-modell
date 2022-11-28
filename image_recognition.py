#%% import libs and get data

import keras.layers as Layers
import keras.models as Models
import keras.utils as Utils
from keras.utils.vis_utils import model_to_dot
import os
import matplotlib.pyplot as plot
import cv2
import numpy as np
from sklearn.utils import shuffle
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


def get_images(directory):
    label = 0
    Labels = []
    Images = []

    for labels in os.listdir(directory):
        if labels == 'buildings':
            label = 0
        elif labels == 'forest':
            label = 1
        elif labels == 'glacier':
            label = 2
        elif labels == 'mountain':
            label = 3
        elif labels == 'sea':
            label = 4
        elif labels == 'street':
            label = 5

        for image_file in os.listdir(
                directory + labels):
            if image_file.endswith(('.jpg', '.png', 'jpeg')):
                image = cv2.imread(directory + labels + r'/' + str(image_file))
                image = cv2.resize(image, (
                    150, 150))
                Labels.append(label)
                Images.append(image)

    return shuffle(Images, Labels, random_state=864925937)


def get_class_label(class_code):
    labels = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}

    return labels[class_code]


train_images, train_labels = get_images('./images/seg_train/seg_train/')

train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images, test_labels = get_images('./images/seg_test/seg_test/')
test_images = np.array(test_images)
test_labels = np.array(test_labels)

train_images = train_images / 255.0
test_images = test_images / 255.0

print("Shape of train images:", train_images.shape)
print("Shape of train labels:", train_labels.shape)

print("Shape of test images:", test_images.shape)
print("Shape of test labels:", test_labels.shape)
# %% exploring the dataset

class_names_list = list(dict.fromkeys(train_labels))

_, train_counts = np.unique(train_labels, return_counts=True)
_, test_counts = np.unique(test_labels, return_counts=True)
pd.DataFrame({'train': train_counts,
              'test': test_counts},
             index=class_names_list
             ).plot.bar()
plt.show()

plt.pie(train_counts,
        explode=(0, 0, 0, 0, 0, 0),
        labels=class_names_list,
        autopct='%1.1f%%')
plt.axis('equal')
plt.title('Proportion of the observed categories')
plt.show()


def display_examples(images, labels, title):
    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(title, fontsize=25)
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(get_class_label(labels[i]))
    plt.show()


display_examples(train_images, train_labels, "Examples of the images of the dataset")

# %% building of the model

model = Models.Sequential([
    Layers.Conv2D(128, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    Layers.MaxPooling2D(2, 2),
    Layers.Conv2D(64, (3, 3), activation='relu'),
    Layers.MaxPooling2D(2, 2),
    Layers.Conv2D(64, (3, 3), activation='relu'),
    Layers.MaxPooling2D(2, 2),
    Layers.Conv2D(32, (3, 3), activation='relu'),
    Layers.MaxPooling2D(2, 2),
    Layers.Flatten(),
    Layers.Dense(256, activation="relu"),
    Layers.Dense(6, activation="softmax")
])

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

model.summary()
SVG(model_to_dot(model).create(prog='dot', format='svg'))
Utils.plot_model(model, to_file='model3.png', show_shapes=True)

# %% train model

trained = model.fit(train_images, train_labels, epochs=25, batch_size=128, validation_split=0.3)

# %% display training accuracy and loss

plot.plot(trained.history['accuracy'])
plot.plot(trained.history['val_accuracy'])
plot.title('Model accuracy')
plot.ylabel('Accuracy')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.show()

plot.plot(trained.history['loss'])
plot.plot(trained.history['val_loss'])
plot.title('Model loss')
plot.ylabel('Loss')
plot.xlabel('Epoch')
plot.legend(['Train', 'Test'], loc='upper left')
plot.show()
# %% evaluate model on test data

test_images = np.array(test_images)
test_labels = np.array(test_labels)
model.evaluate(test_images, test_labels)
# %% get pred data

pred_images, no_labels = get_images('./images/seg_pred/')
pred_images = np.array(pred_images)
# %% showing prediction probability examples

fig = plot.figure(figsize=(30, 30))
outer = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.2)

for i in range(25):
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
    rnd_number = randint(0, len(pred_images))
    pred_image = np.array([pred_images[rnd_number]])
    pred_class = get_class_label((np.argmax(model.predict(pred_image), axis=-1)[0]))
    pred_prob = model.predict(pred_image).reshape(6)
    for j in range(2):
        if (j % 2) == 0:
            ax = plot.Subplot(fig, inner[j])
            ax.imshow(pred_image[0])
            ax.set_title(pred_class)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
        else:
            ax = plot.Subplot(fig, inner[j])
            ax.bar([0, 1, 2, 3, 4, 5], pred_prob)
            fig.add_subplot(ax)

fig.show()
# %% error analysis - showing mislabeled examples

predictions = model.predict(test_images)
pred_labels = np.argmax(predictions, axis=1)


def print_mislabeled_images(test_images, test_labels, pred_labels):
    valid_predictions = (test_labels == pred_labels)
    mislabeled_indices = np.where(valid_predictions == 0)
    mislabeled_images = test_images[mislabeled_indices]
    mislabeled_labels = pred_labels[mislabeled_indices]
    title = "Examples of mislabeled images"

    display_examples(mislabeled_images, mislabeled_labels, title)


print_mislabeled_images(test_images, test_labels, pred_labels)
# %% error analysis with confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sn

class_names_string = []

for label in class_names:
    class_names_string.append(label)

confusion_matrix = confusion_matrix(test_labels, pred_labels)
ax = plt.axes()
sn.heatmap(confusion_matrix, annot=True,
           annot_kws={},
           xticklabels=class_names_string,
           yticklabels=class_names_string, ax=ax)
ax.set_title('Confusion matrix')
plt.show()
# %% saving model

model.save('model10')
# %% saving model in h5 format

model.save('model10.h5')
