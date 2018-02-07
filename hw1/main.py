# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:40:44 2017

@author: ComPeter
"""

import numpy as np
import time
import sys
import PLA
from collections import defaultdict

#tested and seems ok
def get_data():
    #open file
    with open("letter-recognition.data") as file:
        total_x = []
        total_y = []
        #go line by line, removing extraneous info, splitting into a 17d array, append it to total
        for line in file:
            line = line.strip("\n")
            line = line.split(",")
            total_x.append(line[1:])
            total_y.append(line[0])
    #split into train and test
    train_x = total_x[:15000]
    train_y = total_y[:15000]
    test_x = total_x[15000:]
    test_y = total_y[15000:]
    # convert to np arrays
    train_x = np.array(train_x).astype(np.float)
    train_y = np.array(train_y, dtype=str)
    test_x = np.array(test_x).astype(np.float)
    test_y = np.array(test_y, dtype=str)
    #scale data
    train_x, test_x = scale_data(train_x, test_x)
    
    return train_x, train_y, test_x, test_y
 
#tested and seems ok
def scale_data(train_x, test_x):
    #get means of training data
    means = np.mean(train_x)
    #subtract the means from the data
    train_x = np.subtract(train_x, means)
    test_x = np.subtract(test_x, means)
    
    #get variances of training data
    variances = np.var(train_x)
    #divide the variances into the data
    train_x = np.divide(train_x, variances)
    test_x = np.divide(test_x, variances)
    
    #normalize example length
    for row in train_x:
        magnitude = np.linalg.norm(row)
        row = np.divide(row, magnitude)
    for row in test_x:
        magnitude = np.linalg.norm(row)
        row = np.divide(row, magnitude)
        
    return train_x, test_x

#tested and seems ok
def reclassify_y_ova(y, letter):
    len_y = len(y)
    new_y = []
    #loop through train_y
    for i in range(len_y):
        #adding new entries to +/- 1 if they match the supplied letter
        if (y[i].lower() == letter.lower() ):
            new_y.append(1)
        else:
            new_y.append(-1)
    new_y = np.array(new_y, np.float64)
    return new_y

#tested and seems ok
def add_bias(x):
    #insert a 1 into x[0] for all of the training data for the bias term
    x = np.insert(x, 0, 1, axis=1)
    return x

#tested and seems ok
def create_train_subsample(train_x, train_y, num_train):
    new_train_x=[]
    new_train_y=[]
    #only get as many examples as I ask for
    for i in range(num_train):
        #generate a random number and get the data from the training set
        ran = np.random.randint(0, len(train_x))
        #slice off the target variable and append it to the array for the x, keep only the target for the y
        new_train_x.append(train_x[ran])
        new_train_y.append(train_y[ran])
    #turn it into a numpy array and return it
    new_train_x = np.array(new_train_x, np.float64)
    new_train_y = np.array(new_train_y, np.str)
    return new_train_x, new_train_y

#tested and seems ok
def compute_accuracy(test_y, pred_y, confusion_bool):
    #get the lengths of the test and prediction, if they don't match something went wrong and exit
    lenTest = len(test_y)
    lenPred = len(pred_y)
    if(lenTest!=lenPred):
        print("Something went wrong, there is a length mismatch in the test/pred, exiting...")
        sys.exit(1)
    #to keep track of the numbers right and wrong
    numWrong = 0
    numRight = 0
    pla_bool = False
    knn_bool= False
    #if i am going to make a confusion matrix
    if(confusion_bool):
        confusion_matrix = defaultdict(int)
    #since PLA converts test y to 0's or 1's i check if I'm running computer acc on a pla test/pred
    if(test_y[0] == -1 or test_y[0] == 1 ):
        pla_bool = True
    #if it's not an +/- 1, im comparing knn in the form of strings, not numbers, so i need to calculate it differently
    else:
        knn_bool = True
        
    for i in range(lenTest):
        #if we are computing the accuracy for a PLA problem
        if(pla_bool):
            if(test_y[i] > 0 and pred_y[i] > 0):
                confusion_matrix["pyay"] += 1
                numRight += 1
            elif(test_y[i] > 0 and pred_y[i] < 0):
                confusion_matrix["pnay"] += 1
                numWrong += 1
            elif(test_y[i] < 0 and pred_y[i] > 0):
                confusion_matrix["pyan"] += 1
                numWrong +=1
            else:
                confusion_matrix["pnan"] += 1
                numRight +=1
        #else if we are computing the accuracy for a knn problem
        elif(knn_bool):
            #get the letters from the test and prediction
            test_y_val = test_y[i].lower()
            pred_y_val = pred_y[i].lower()
            #combine them and add them to the confusion matrix
            combined = test_y_val + pred_y_val
            confusion_matrix[combined] += 1
            #also update the number of right/wrong
            if( test_y_val == pred_y_val ):
                numRight += 1
            else:
                numWrong += 1
        #else something went incredibly wrong and we should exit
        else:
            print("Something went terribly wrong, it is neither a PLA nor KNN problem, exiting...")
            sys.exit(1)
                
    numRight = float(numRight)
    lenTest = float(lenTest)
    print("The number right is:" + str(numRight) + ", the number wrong is: " + str(numWrong) + ", the percentage is: " + str(numRight/lenTest))
    if(confusion_bool):
        print("The confusion matrix is:")
        print(confusion_matrix)
    return numRight/lenTest

#still work to do
def main():
    #get the data from the file, and scale/normalize it
    train_x, train_y, test_x, test_y = get_data()
    #list of the different sizes i need to train
    num_train = [100, 1000, 2000, 5000, 10000, 15000]
    #loop through the different sizes of training sets
    for i in num_train:
        #run pla on every letter, this is OVA
        #generate trainin subset for all the letters I need
        sub_train_x, sub_train_y = create_train_subsample(train_x, train_y, i)
        #mark the time
        start = time.time()
        rollingAccuracy = 0.0
        for letter in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']:
            #create a set of training data with a bias term, adjust y to be +/- 1
            temp_train_x = add_bias(sub_train_x)
            temp_train_y = reclassify_y_ova(sub_train_y, letter)
            temp_test_x = add_bias(test_x)
            temp_test_y = reclassify_y_ova(test_y, letter)
            #create PLA object
            pla_object = PLA(temp_train_x, temp_train_y, temp_test_x, temp_test_y)
            #train the PLA object
            pla_object.train_pocket()
            #get the predictions from the PLA object
            pred_y = pla_object.test_pla()
            #get the accuracy
            acc = compute_accuracy(temp_test_y, pred_y, True)
            #add it to my total accuracy
            rollingAccuracy += acc
        end = time.time()
        duration = end-start
        rollingAccuracy /= 26
        #mark the end time, get the duration, get the average over the whole alphabet, print the information
        print("For pla with num_train: " + str(i) + 
              " averaged over all 26 letters is: " + str(rollingAccuracy) + 
              " the time to execute is: " + str(duration))
        
        #numpy arrays are referential, so any edits I do happen in the real object, not a copy of the object in a functions stack
        #this means i need to regenerate the training set to "undo" the edits i needed for OVA
        train_x, train_y = create_train_subsample(train_x, train_y, i)
        #k-NN values as range, 1, 3, 5, 7, 9
        for k in range(1,10,2):
            #get the time
            start = time.time()
            #run the algo
            pred_y = test_knn(train_x, train_y, test_x, k)
            #mark the time
            end = time.time()
            #check the accuracy
            acc = compute_accuracy(test_y, pred_y)
            #compute the duraction
            duration = end-start
            print("For knn with num_train: " + str(i) + 
              " the number of nearest neighbors is: " + str(k) + 
              " the accuracy is: " + str(acc) +
              " the time to execute is: " + str(duration))
        #regenerate training and testing data   
        train_x, train_y = create_train_subsample(training_data, i)
        test_x, test_y = createTest_xAndy(testing_data)
        #mark condensing time
        start = time.time()
        #condense
        condensed_idx = condense_data(train_x, train_y)
        #end condense time and calculate
        end = time.time()
        condense_time = end - start
        #adjust x and y using the condensed indices
        adjusted_x, adjusted_y = shortenTraining(train_x, train_y, condensed_idx)
        #mark the time
        start = time.time()
        #run the condensed one
        pred_y = test_knn(adjusted_x, adjusted_y, test_x, k)
        #end time, duration, calc accuracy
        end = time.time()
        duration = end - start
        acc = compute_accuracy(test_y, pred_y)
        print("For condensed knn with num_train: " + str(i) + 
              " the accuracy is: " + str(acc) +
              " the time to condense is: " + str(condense_time) +
              " the time to execute is: " + str(duration) +
              " the len of condensed is:" + str(len(condensed_idx)))
 
    
    #acc = compute_accuracy(test_y, pred_y)

#this is a type of crude unit testing to check my functions
def testing():
    #testing the get_data function
    print("TESTING THE get_data FUNCTION:")
    train_x, train_y, test_x, test_y = get_data()
    print("train_x's shape is: " + str(train_x.shape))
    print("train_y's shape is: " + str(train_y.shape))
    print("test_x's shape is: " + str(test_x.shape))
    print("test_y's shape is: " + str(test_y.shape))
    
    #testing the scale_data function
    print("TESTING THE scale_data FUNCTION:")
    print("train_x[1] before scaling is:" + str(train_x[1]))
    print("test_x[1] before scaling is:" + str(test_x[1]))
    train_x, test_x = scale_data(train_x, test_x)
    print("train_x[1] after scaling is:" + str(train_x[1]))
    print("test_x[1] after scaling is:" + str(test_x[1]))
    #test/train x are scaled in get_data, so this rescales them again (this is not useful except as a check that it works)
    
    #testing the reclassify_y_ova function
    print("TESTING THE reclassify_y_ova FUNCTION:")
    print("the first 10 train_y before are:" + str(train_y[0:10]))
    print("the first 10 test_y before are:" + str(test_y[0:10]))
    train_y = reclassify_y_ova(train_y, 'a')
    test_y = reclassify_y_ova(test_y, 'a')
    print("the first 10 train_y after are:" + str(train_y[0:10]))
    print("the first 10 test_y after are:" + str(test_y[0:10]))
    
    #testing the add_bias function
    print("TESTING THE add_bias FUNCTION:")
    print("train_x's shape before is: " + str(train_x.shape))
    print("test_x's shape before is: " + str(test_x.shape))
    train_x = add_bias(train_x)
    test_x = add_bias(test_x)
    print("train_x's shape after is: " + str(train_x.shape))
    print("test_x's shape after is: " + str(test_x.shape))
    
    #testing create_train_subsample
    print("TESTING THE create_train_subsample FUNCTION:")
    print("train_x's shape before is: " + str(train_x.shape))
    print("train_y's shape before is: " + str(train_y.shape))
    train_x, train_y = create_train_subsample(train_x, train_y, 1000)
    print("train_x's shape after is: " + str(train_x.shape))
    print("train_y's shape after is: " + str(train_y.shape))
    
    #testing compute_accuracy for pla
    print("TESTING THE compute_accuracy FUNCTION FOR PLA WITH CONFUSION MATRIX (expected accuracy 50%):")
    pla_pred = [-1, -1, 1, 1, -1, -1, 1, 1]
    pla_test = [1, -1, 1, -1, 1, -1, 1, -1]
    compute_accuracy(pla_test, pla_pred, True)
    
    #testing compute_accuracy for knn
    print("TESTING THE compute_accuracy FUNCTION WITH CONFUSION MATRIX (expected accuracy 70%):")
    knn_pred = ["a","a","b","c","d","a","a","b","c","d"]
    knn_test = ["a","a","a","c","b","d","a","b","c","d"]
    compute_accuracy(knn_test, knn_pred, True)
    

if __name__ == "__main__":
    #main()
    
    testing()
    
    
    
