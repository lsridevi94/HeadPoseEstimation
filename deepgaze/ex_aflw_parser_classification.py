#!/usr/bin/env python

##
# Massimiliano Patacchiola, Plymouth University 2016
#
# File with a parser for the Tugraz "Annotated Facial Landmarks in the Wild" dataset.
# https://lrs.icg.tugraz.at/research/aflw/
#
# I wrote a python code to access the sqlite database, it can be found on the  official
# page of the dataset (it requires free registration):
# https://lrs.icg.tugraz.at/research/aflw/downloads.php
# 
# This file loads two numpy files (dataset.npy and label.npy), which
# must be created in advance.
# If the files are not found the script looks for a folder with
# the picture to load and the csv file containing: 
# image_name, roll, pitch, yaw 
#
# The two numpy files contain:
# dataset.npy: the images of the dataset in a 64x64 format
#  arranged in in a matrix of shape=(tot_imgs,64*64)
# label.npy: the roll,pitch,yaw data of the face 
#  arranged in a matrix of shape=(tot_imgs, 3)
#
# The script produces a single sanitized pickle file with
# 3 datasets and 3 label arrays (training, validation, test).
# The pickle file will load the arrays in the training script.

import os.path
import numpy
import cv2
import csv
from six.moves import cPickle as pickle

#The path for the dataset images and for the csv file with
# the label information.
dataset_npy_path = "dataset.npy"
label_npy_path = "label.npy"
dataset_image_path = "../HeadPoseImageDatabase/training"
label_csv_path = "../HeadPoseImageDatabase/training/prima_label.csv"

#If the dataset numpy file do not exist it create it
if(os.path.isfile(label_npy_path)==False or os.path.isfile(dataset_npy_path)==False):

    #Saving the file names in a list
    image_list = list()
    with open(label_csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            image_list.append(row[0])

    #Loading the label
    label = numpy.genfromtxt(label_csv_path, delimiter=',', skip_header=1, usecols=(3,4), dtype=numpy.str)
    label_row, label_col = label.shape
    print(label.shape)
    print(label[0][:])

    dataset_row = label_row
    dataset_col = 64 * 64 #the size of the image
    dataset = numpy.zeros((dataset_row, dataset_col), dtype=numpy.int8)

    row_counter = 0
    for image_name in image_list:
        if image_name !='path':
            image_path = image_name
            image = cv2.imread(image_path, 0) #load in greyscale
            dataset[row_counter] = image.reshape(1, -1)
            row_counter += 1 

    #Saving the numpy array to files
    numpy.save("label", label)
    numpy.save("dataset", dataset)
    print(dataset.dtype)

print("Loading the numpy file with the dataset and the label")

#Load the dataset and labels
#The dataset contains integers
#in the range -127/+127
label = numpy.load("./label.npy")
dataset_int = numpy.load("./dataset.npy")


print("Rescaling the image values from range -127/+127 to range -1/+1")

#Applying min-max scaling to the dataset
#After this step the data will be in the
# range -1/+1
dataset = dataset_int.astype(dtype=numpy.float32)
dataset /= 127 #dividing by the max value

# #Roll
# max_roll = numpy.amax(label[:,0])
# min_roll = numpy.amin(label[:,0])
# std_roll = numpy.std(label[:,0])
# mean_roll = numpy.mean(label[:,0])
# print("\n")
# print("===  ROLL  === \n" +  "mean: " + str(mean_roll) + "\n" + "std: " + str(std_roll) + "\n" + "max: " + str(max_roll) + "\n" + "min: " + str(min_roll) + "\n")

#Pitch
#max_pitch = numpy.amax(label[:,0])
#min_pitch = numpy.amin(label[:,0])
#std_pitch = numpy.std(label[:,0])
#mean_pitch = numpy.mean(label[:,0])
#print("=== PITCH === \n" +  "mean: " + str(mean_pitch) + "\n" + "std: " + str(std_pitch) + "\n" + "max: " + str(max_pitch) + "\n" + "min: " + str(min_pitch) + "\n")
#
##Yaw
#max_yaw = numpy.amax(label[:,1])
#min_yaw = numpy.amin(label[:,1])
#std_yaw = numpy.std(label[:,1])
#mean_yaw = numpy.mean(label[:,1])
#print("===  YAW  === \n" +  "mean: " + str(mean_yaw) + "\n" + "std: " + str(std_yaw) + "\n" + "max: " + str(max_yaw) + "\n" + "min: " + str(min_yaw) + "\n")

#Temporary append the label to the dataset to shuffle the data
#Temporary append the label to the dataset to shuffle the data
data = numpy.append(dataset, label, axis=1)
#Shuffle the row to randomize the data
#print(data.dtype)
numpy.random.shuffle(data)

#Separating the label from the dataset
dataset = numpy.float32(data[:,0:4096])
label = data[:,4096:4098]
print(dataset.dtype)
#Creating the trainin and test datasets
# #Training dataset is 60% of the total
# row, col = dataset.shape
# cut_80 = int(row * 0.8) #take 80% of the total
# cut_10 = (row - cut_80)/2 #split the remaining 20%
training_dataset = numpy.copy(dataset[0:2604,:]) 
training_label = numpy.copy(label[0:2604,:]) 
# #Validation dataset is 10% of the total
# validation_dataset = numpy.copy(dataset[cut_80:cut_80+cut_10,:])
# validation_label = numpy.copy(label[cut_80:cut_80+cut_10,:])
# #Test dataset is 10% of the total
test_dataset = numpy.copy(dataset[2604:2791,:])
test_label = numpy.copy(label[2604:2791,:])

#saving the sanitized dataset in a pickle file
pickle_file = 'aflw_dataset.pickle'
print("Saving the dataset in: " + pickle_file)
print("... ")
try:
    f = open(pickle_file, 'wb')
    save = {
        'training_dataset': training_dataset,
        'training_label': training_label,       
        'test_dataset': test_dataset,
        'test_label': test_label        
        }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

print("The dataset has been saved and it is ready for the training! \n")



