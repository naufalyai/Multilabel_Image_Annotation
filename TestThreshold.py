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
from sklearn.metrics import matthews_corrcoef
import copy
def accuracy_metric(y_true, y_pred):
    true = 0
    false = 0
    acc = 0
    rec = 0
    print('Total Label : 3')

    for i in range(y_true.shape[0]):
        print('=============================================================================================')
        print('For Data : ', i)
        totlab = np.sum(y_pred[i])

        true = 0
        false = 0
        for j in range(y_true.shape[1]):
            if(y_pred[i,j] == 1):

                if(y_pred[i,j] == y_true[i,j]):
                    true+=1
                else :
                    false +=1
        if ((np.sum(y_true[i]) >= totlab) & (totlab!=0)):
            if(true <=totlab):
                accu = (true / totlab)
            else:
                accu = (totlab/totlab)
        else:
            if (totlab == 0):
                accu = 0
            else:
                accu = (true / np.sum(y_true[i]))
        recallin = (true / np.sum(y_true[i]))
        # print('Accu : ',accu)
        acc += accu
        # print('Akur : ', acc)
        rec += recallin


        print('Total Label : ',totlab)
        print('Prediction : ', y_pred[i])
        print('Actual : ', y_true[i])
        print('Actual Label : ', np.sum(y_true[i]))
        print('Correct Label : ',true, ' from ', np.sum(y_true[i]), ' labels')
        # print('Akur : ', acc)
    akurasinya = acc/y_pred.shape[0]
    recallnya = rec/y_pred.shape[0]
    return (akurasinya,recallnya)

def openTH(filename):
    file = open(filename, "r")
    best_threshold = []
    for i in file:
        best_threshold.append(float(i))
    return(best_threshold)

model_name = './saved_models/keras_nus_trained_model_skenario1_50.h5'
model = keras.models.load_model(model_name)
print('Running model : ', model_name)
f_in = h5py.File('./test/TESTV3.hdf5', 'r')
x_train = f_in['test_img']
print(x_train.shape[0])
y_train = f_in['test_labels']

best_threshold = openTH("./dummy2/TOPTH_NOBNDO75.txt")
print("Best Threshold: ",best_threshold)
preds = model.predict(x_train)
# print("initial preds: ", preds[1])
y_pred = np.array(
    [[1 if preds[i, j] >= best_threshold[j] else 0 for j in range(y_train.shape[1])] for i in range(len(y_train))])

df = pandas.DataFrame(y_pred)
df.to_excel('./dummy2/TestTOPTH_NOBNDO75v2.xlsx')
akurasi, recall2 = accuracy_metric(y_train,y_pred)
print('Manually Accuracy : ',akurasi*100,' %')
print('Micro Average F1 Score : ',f1_score(y_train, y_pred, average='micro'))
print('Micro Average Precision : ',precision_score(y_train, y_pred, average='micro'))
print('Micro Average Recall : ',recall_score(y_train, y_pred, average='micro'))

