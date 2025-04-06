import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2


# path to file
dataFile = './cifar10/test_batch'
metaFile = './cifar10/batches.meta'

def plottingModel():
    # Plot accuracy
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.show()

# loading CIFAR-10 data & metadata
def loadData(fileName, normalize=True):
    with open(fileName, 'rb') as f:
        dataDict = pickle.load(f, encoding='bytes')
        if b'data' in dataDict:
            data = dataDict[b'data']
            labels = dataDict[b'labels']
            # reshaping & transposing to correct dimensions
            images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            # normalizing pixel values between range [0, 1]
            if normalize:
                images = images / 255.0
            return images, np.array(labels)
        if b'label_names' in dataDict:
            return [name.decode('utf-8') for name in dataDict[b'label_names']]

# load data & metadata
images, labels = loadData(dataFile)
labelNames = loadData(metaFile)
imageShape = images[0].shape
print(f"Data shape: {images.shape}, Labels shape: {labels.shape}")
print(f"Pixel value range: Min={images.min()}, Max={images.max()}")
print(f"Label Names: {labelNames}")
print(f'SHAPE IMAGE: {imageShape}')
# print(f'Image example:{images[0]}')


def displayImages(images, labels, labelNames, numImages=5):
    """displayes first few images w/ labels
    """
    for i in range(numImages):
        plt.imshow((images[i] * 255).astype('uint8'))  # Scale back to [0, 255] for display
        plt.title(labelNames[labels[i]])
        plt.xticks([])
        plt.yticks([])
        plt.show()

def displayRandomGrid(images, labels, labelNames, rows=5, cols=5):
    """
    displays a random grid of imgs
    """
    fig = plt.figure(figsize=(10, 10))
    randomIdx = np.random.randint(0, len(images), rows * cols)
    for i, idx in enumerate(randomIdx):
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow((images[idx] * 255).astype('uint8'))  # Scale back to [0, 255] for display
        plt.title(labelNames[labels[idx]])
        plt.xticks([])
        plt.yticks([])
    plt.show()

# # displays:
# displayImages(images, labels, labelNames)
# displayRandomGrid(images, labels, labelNames)

uniqueLabels= int(len(np.unique(labels)))
labelsCategorical = to_categorical(labels, num_classes=uniqueLabels)
print("One hot encoded categorical label example:",labelsCategorical[0])


## MAKE CNN:
# model=Sequential()
# model.add(Conv2D(128, kernel_size=3, activation="relu", input_shape=imageShape))
# model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Conv2D(64, kernel_size=3, activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten()) #connects convolution and dense layer
# model.add(Dense(10, activation='softmax')) #10 probs added to probabilities
model = Sequential()
model.add(Conv2D(128, kernel_size=3, activation="relu", input_shape=imageShape))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
#compile model using accuracy
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#splitting data
X_train, X_test, y_train, y_test = train_test_split(images, labelsCategorical, random_state=0, train_size=.7, shuffle=True)
#introducing early stopping for overfitting
earlyStopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
#training model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[earlyStopping])
test_loss, test_acc = model.evaluate(images, labelsCategorical, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")
print("MAX Validation Accuracy", max(list(history.history['val_accuracy'])))
