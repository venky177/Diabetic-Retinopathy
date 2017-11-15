import numpy as np
import cv2
import glob
import pickle
import csv
import os
def load_train_label(train_path, label_file, image_size, classes):
    images = []
    labels = []
    ids = []
    cls = []

    label_fp = open(label_file,'r')
    label_arr = csv.reader(label_fp,delimiter=' ')
    label_dict = {}
    for i in label_arr:
        key,value = i[0].split(",")
        label_dict[key]=value

    print('Reading training images')
    files = glob.glob(train_path+"*.jpeg")
    for fl in files[20001:25000]:
        file_name = os.path.basename(fl).split(".")[0]

        image = cv2.imread(fl)
        image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
        images.append(image)
        label = np.zeros(len(classes))
        label[int(label_dict[file_name])] = 1.0
        labels.append(label)
        flbase = os.path.basename(fl)
        ids.append(flbase)
        cls.append(label_dict[file_name])
    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    return images, labels, ids, cls

train_path="/home/venky/Downloads/train/"#'/home/venky/DR/cv-tricks.com-master/Tensorflow-tutorials/tutorial-2-image-classifier/training_data'
test_path='/home/venky/DR/cv-tricks.com-master/Tensorflow-tutorials/tutorial-2-image-classifier/testing_data'
label_file = "/home/venky/Downloads/trainLabels.csv"
image_size = 256
classes = [0,1,2,3,4]
i = 4
images, labels, ids, cls = load_train_label(train_path, label_file, image_size, classes)
np.save('images_256_'+str(i)+'.npy',images)
#np.save('labels'+str(i)+'.npy',labels)
#np.save('fnames'+str(i)+".npy",ids)

'''lbl = np.array()
for i in [2,3,4,5,6]:
    lbl=np.concatenate((lbl,np.load('labels'+str(i)+'.npy')))
cls = np.array([np.where(x==1)[0][0] for x in lbl])'''