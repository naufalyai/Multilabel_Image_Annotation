import csv
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, confusion_matrix, f1_score, accuracy_score, matthews_corrcoef, hamming_loss, precision_recall_curve, average_precision_score
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


def readcsv(filename):

    with open(filename, 'rU') as p:
        a = [list(map(int,rec)) for rec in csv.reader(p, delimiter=';')]
        p.close()
    return a
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
            if(totlab == 0):
                accu = 0
            else:
                accu = (true / np.sum(y_true[i]))
        recallin = (true / np.sum(y_true[i]))
       
        acc += accu
       
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
def precisionrecall(y_true, y_pred):

    prec = []
    rec = []
    confusion = []
    for i in range(y_true.shape[0]):
        # totalab.append(np.sum(y_true[i,:]))
        cm = []
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for j in range(y_true.shape[1]):
           if(y_pred[i,j] == 1):
               if(y_pred[i,j] == y_true[i,j]):
                   tp +=1
               else :
                   fp +=1
           else:
               if(y_true[i,j] == 1):
                   fn +=1
               else:
                   tn +=1
        gt = np.sum(y_true[i,:])
        confusion.append([tp,fp,tn,fn,gt])

        if((tp == 0) & (fp == 0)):
            prec.append(0)
        else:
            prec.append(tp/(tp+fp))
        if(tp == 0 & fn == 0):
            rec.append(0)
        else:
            rec.append(tp/(tp+fn))
    return (np.mean(prec),np.mean(rec), prec, rec, confusion)

f_in = h5py.File('./test/TESTV3.hdf5', 'r')
y_train = f_in['test_labels']
path = "labelScenev2.txt"
path2 = "cekScene.txt"
pathImage = "./test/*.jpg"
labels = loadLabel(path)
ceklab = cekLabel(path2)
classes = ['window','waterfall','water','valley','town','temple','sunset','street','snow','sky','road','reflection','railroad','plants','ocean','night_time','mountain','moon','lake','house','harbor','grass','glacier','garden','frost','clouds','cityscape','buildings','bridge','beach']

a = readcsv('./dummy3/TestNBN15DO5Top5.csv')
a2 = np.asarray(a)
precisiontot = []
recalltot = []
labelprec = []
labelrec = []
prec = dict()
rec = dict()
average_precision = dict()
for i in range(30):
    
    precision = precision_score(y_train[:,i],a2[:,i])
    recall = recall_score(y_train[:,i],a2[:,i])
    labelprec.append([classes[i], precision])
    precisiontot.append(precision)
    recalltot.append(recall)
    labelrec.append([classes[i], recall])
    
labelrec = np.array(labelrec)
labelprec = np.array(labelprec)
precisiontot = np.array(precisiontot)
recalltot = np.array(recalltot)
avg_prec = np.average(precisiontot)
avg_rec = np.average(recalltot)

hammingLoss = hamming_loss(y_train,a2)
akurasi, recall2 = accuracy_metric(y_train,a2)
print('Manually Accuracy : ',akurasi*100,' %')
print('Micro Average Precision : ', precision_score(y_train,a2,average='micro'))
print('Micro Average F1 : ', f1_score(y_train,a2,average='micro'))
print('Micro Average Recall : ', recall_score(y_train,a2,average='micro'))
print('Macro Average Precision : ', precision_score(y_train,a2,average='macro'))
print('Macro Average F1 : ', f1_score(y_train,a2,average='macro'))
print('Macro Average Recall : ', recall_score(y_train,a2,average='macro'))
meanprec,meanrec,precis,recas,confusion = precisionrecall(y_train,a2)
df = pandas.DataFrame(labelprec)
df.to_excel('./dummy3/precision/TestNBN15DO5Top5Precision.xlsx',header=['tag','precision'])
dp = pandas.DataFrame(labelrec)
dp.to_excel('./dummy3/recall/TestNBN15DO5Top5Recall.xlsx',header=['tag','recall'])
dr = pandas.DataFrame(confusion)
dr.to_excel('./dummy3/confusion/TestNBN15DO5Top5Confusion.xlsx',header=['TP','FP','TN','FN','Ground_Truth'])
