import keras
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPool1D, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
import h5py
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, hamming_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import normalize, MinMaxScaler
import copy
from sklearn.utils import class_weight
import pandas


f_in = h5py.File('./train/TRAINFIX.hdf5', 'r')
x_train = f_in['test_img']
y_train = f_in['test_labels']

model_name = './saved_models/keras_nus_trained_model_skenario1_50.h5'
model = keras.models.load_model(model_name)

classes = ['window','waterfall','water','valley','town','temple','sunset','street','snow','sky','road','reflection','railroad','plants','ocean','night_time','mountain','moon','lake','house','harbor','grass','glacier','garden','frost','clouds','cityscape','buildings','bridge','beach']


#################################################
# Evaluate Model
#################################################


print("Evaluating Model")
# x = model.evaluate(x_train, y_train)
# print(x)

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
import copy

temp_preds = model.predict(x_train)
print("initial preds: ", temp_preds[1])
th = 0.5
print("1a. Uniform Threshold =", th)
preds = copy.deepcopy(temp_preds)
preds[preds >= th] = 1
preds[preds < th] = 0
f1 =[]

print("example result")
print("predicted: ", preds[1].astype(int))
print("target:    ", y_train[1])

#################################################
# Finding best threshold
#################################################

print("2. Label Dependant Threshold")

preds = copy.deepcopy(temp_preds)

preds = np.array(preds)
# print(out)
threshold = np.arange(0.1,0.9,0.01)

acc = []
accuracies = []
best_threshold = np.zeros(preds.shape[1])
for i in range(preds.shape[1]):
    y_prob = np.array(preds[:, i])
    for j in threshold:
        y_pred = [1 if prob >= j else 0 for prob in y_prob]
        acc.append(matthews_corrcoef(y_train[:, i], y_pred))
    acc = np.array(acc)
    index = np.where(acc == acc.max())
    accuracies.append(acc.max())
    best_threshold[i] = threshold[index[0][0]]
    acc = []

print("accuracies = ", accuracies)
print("best thresholds = \n", best_threshold)
y_pred = np.array(
    [[1 if preds[i, j] >= best_threshold[j] else 0 for j in range(y_train.shape[1])] for i in range(len(y_train))])
# f1 = []
print("new F1 Score = ", f1_score(y_train, y_pred, average='micro'))
print('Micro Average Precision : ',precision_score(y_train, y_pred, average='micro'))
print('Micro Average Precision : ',recall_score(y_train, y_pred, average='micro'))

randomize = np.random.randint(1,99)
print("example result")
print("predicted: ", y_pred[randomize])
print("target:    ", y_train[randomize])

df = pandas.DataFrame(y_pred)
df.to_excel('./dummy3/TOPTH_BNScratch50.xlsx',header=classes)
np.savetxt('./dummy3/TOPTH_BNScratch50.txt',best_threshold,delimiter=',')
