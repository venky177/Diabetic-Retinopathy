import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD,Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')
import dataset

# class info

classes = [0,1,2,3,4]
num_classes = len(classes)
# image dimensions (only squares for now)
img_size = 256
# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3


label_file = "/home/venky/Downloads/trainLabels.csv"
train_path="/home/venky/Downloads/smallTrain/"#'/home/venky/DR/cv-tricks.com-master/Tensorflow-tutorials/tutorial-2-image-classifier/training_data'
validation_size = .2

data = dataset.read_train_sets_with_labels(train_path,label_file, img_size, classes, validation_size=validation_size)

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_size, img_size, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 30
lrate = 1e-4
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())

X_train = data.train._images
X_valid = data.valid._images

y_train = data.train._labels
y_valid = data.valid._labels

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epochs, batch_size=4)
# Final evaluation of the model
scores = model.evaluate(X_valid, y_valid, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))