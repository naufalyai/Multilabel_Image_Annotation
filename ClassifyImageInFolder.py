import keras
import numpy as np
import h5py
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, precision_score, recall_score
from skimage import io
import glob
import pandas

model_name = '.\saved_models\keras_nus_trained_model_skenario1_10.h5'
path = "labelTest.txt"
path2 = 'testCheck.txt'

classes = ['window','waterfall','water','valley','town','temple','sunset','street','snow','sky','road','reflection','railroad','plants','ocean','night_time','mountain','moon','lake','house','harbor','grass','glacier','garden','frost','clouds','cityscape','buildings','bridge','beach']

model = keras.models.load_model(model_name)

# preds = model.predict(x)
def cekLabel(pathfile):
    ee = []
    with open(pathfile) as infile:
        file_contents = infile.readlines()
        for f in file_contents:
            d = f.strip('\t')
            e = d.split()
            for a in e:
                ee.append(int(a))
    return ee
def showClass(y_pred):
    labels = []
    for i in range(30):
        # print(y_pred[0,i])
        if (y_pred[0,i] == 1):
            labels.append(classes[i])
    return labels
def loadLabel(pathfile):
    ee = []
    with open(pathfile) as infile:
        file_contents = infile.readlines()
        for f in file_contents:
            d = f.strip('\t')
            e = d.split()
            ff = []
            for a in e:
                ff.append(int(a))
            ee.append(ff)
    return ee
ceklab = cekLabel(path2)
y_true = loadLabel(path)
model = keras.models.load_model(model_name)
pathImage = "./test/*.jpg"
addrs = glob.glob(pathImage)
predicition = []
labeli = []
import os
dest = 'test'
for i in range(len(addrs)):
    if(ceklab[i] == 1):
        # print(i)
        adres = os.path.join(dest,(str(i)+'.jpg'))
        x = io.imread(adres)
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)

        top_k_indx = np.argsort(preds[0])[:-(5+ 1):-1]

        aaa = np.zeros([1, 30], dtype=int)
        y_pred = []
        y_lab = []
        for j in top_k_indx:

            aaa[0][j] = 1
        y_pred.append(aaa[0])
        y_pred = np.array(aaa)
        # y_truth = np.array(y_true[i])
        y_lab.append(y_true[i])
        y_lab = np.expand_dims(y_true[i],axis=0)
        # print(y_pred.shape)
        predicition.append(aaa[0])
        # print(y_lab.shape)
        lab = showClass(aaa)
        truelab = showClass(y_lab)
        labeli.append([adres,lab,truelab])
        print('Name : ', adres)
        print('Prediction : ',lab)
        print('Actual : ', truelab)

df = pandas.DataFrame(predicition)
df.to_excel('./dummy3/TestBN10DO25Top5.xlsx',header=classes)
df = pandas.DataFrame(labeli)
df.to_excel('./dummy3/TestBN10DO25LabelTop5.xlsx',header=['Image_Name','Prediction','Target'])