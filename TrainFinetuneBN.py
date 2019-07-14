from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Activation, Dropout
from keras.models import Model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
import h5py
import os
import keras.metrics as met
import keras
import functools

num_classes = 30
f_in = h5py.File('./train/TRAINFIX2.hdf5', 'r')
x_train = f_in['test_img']
y_train = f_in['test_labels']
print(x_train.shape[1:])
print(y_train.shape)
batch_size = 15
epochs = 100
# load VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# define number of class and output layer
new_class = 30
x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.25)(x)
predictions = Dense(new_class, activation='sigmoid')(x)
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_nus_trained_modelarchitecturePretrainedTOP3500.h5'
weight_name = 'WeightTOP5BN200.h5'
checkpoint_name = "./saved_models/checkpoint_TOP5BN200.h5"

# combine VGG16 and new define output layer
new_model = Model(inputs=model.input, outputs=predictions)

for layer in model.layers:
    layer.trainable = False


new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['top_k_categorical_accuracy'])

new_model.summary()

checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_acc', verbose=1, save_best_only=True,
                             save_weights_only=False,
                             mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

new_model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.3,

                  callbacks=[checkpoint],
                  verbose=1,
                  shuffle="batch"
                  )

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
weightpath = os.path.join(save_dir, weight_name)
new_model.save(model_path)
new_model.save_weights(weightpath)


print('Saved trained model at %s ' % model_path)

