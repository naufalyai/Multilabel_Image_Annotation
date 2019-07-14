from __future__ import print_function
import keras
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPool1D, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
import h5py
import math

import os

train_samples =141918
val_samples = 1658

num_classes = 30
f_in = h5py.File('./train/TRAINFIX2.hdf5', 'r')
x_train = f_in['test_img']
y_train = f_in['test_labels']

batch_size = 5

epochs = 10
data_augmentation = False
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_nus_trained_model_skenario1_15.h5'
weight_name = 'keras_nus_trained_weight_skenario1_15.h5'


# img_width, img_height = 224, 224
model = Sequential()
model.add(Conv2D(64, (3,3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3), padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))


model.add(Conv2D(128, (3,3), padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(128, (3,3),padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(256, (3,3), padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(256, (3,3),padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(256, (3,3),padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(512, (3,3), padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(512, (3,3),padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(512, (3,3),padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(512, (3,3), padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(512, (3,3),padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(512, (3,3),padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(512, (3,3), padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(512, (3,3),padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Conv2D(512, (3,3),padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(GlobalAveragePooling2D())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

opt = keras.optimizers.Adam(lr=1e-3)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['top_k_categorical_accuracy'])
print(model.summary())
checkpoint = ModelCheckpoint("./saved_models/skenario1_2.h5", monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=False,
                             mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.3,
              callbacks=[checkpoint],
              shuffle=True
              )

else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=True,  # divide each input by its std
        zca_whitening=True,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    # datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=train_samples // batch_size,
                        epochs=epochs)


# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
weightpath = os.path.join(save_dir,weight_name)
model.save(model_path)
model.save_weights(weightpath)

print('Saved trained model at %s ' % model_path)
