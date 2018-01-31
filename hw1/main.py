# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:40:44 2017

@author: ComPeter
"""

import numpy as np
import time
from sklearn import preprocessing

def getData():
    #get data
    
    #open file
    with open("letter-recognition.data") as file:
        total = []
        #go line by line, removing extraneous info, splitting into a 17d array, append it to total
        for line in file:
            line = line.strip("\n")
            line = line.split(",")
            total.append(line)
    #a normal python array to store the slices that represent my training and my testing
    training = total[:15000]
    testing = total[15000:]
    #split into train and test
    # convert to 2 np arrays, one is 15000 x 17, the other is 5000 x 17
    training_data = np.array(training)
    testing_data = np.array(testing)
    
    #scale data
    
    
    return None
    
def scaleData():
    #magnitude
    
    #variance
    
    #PCA
    
    return None

def OVAadjustY(y, letter):
    #loop through train_y
    for i in range(len(y)):
        #adjusting the entries to +/- 1 if they match the supplied letter
        if (y.item(i).lower() == letter.lower() ):
            y[i] = 1
        else:
            y[i] = -1
            y = np.array(y, np.float64)
    return y

def adjustX(x):
    #insert a 1 into xo for all of the training data for the bias term
    x = np.insert(x, 0, 1, axis=1)
    return x

def compute_accuracy(test_y, pred_y):
    lenTest = len(test_y)
    lenPred = len(pred_y)
    numWrong = 0
    numRight = 0
    if(lenTest!=lenPred):
        return None
    #since PLA converts test y to 0's or 1's i check if I'm running computer acc on a pla test/pred
    if( test_y.item(0) == -1 or test_y.item(0) == 1 ):
        for i in range(len(test_y)):
            if(test_y[i]*pred_y[i] > 0):
                numRight += 1
            else:
                numWrong += 1
    #if it's not an +/- 1, im comparing knn in the form of strings, not numbers, so i need to calculate it differently
    else:
        for i in range(len(test_y)):
            if( test_y.item(i).lower() == pred_y.item(i).lower()  ):
                numRight += 1
            else:
                numWrong += 1
                
    numRight = float(numRight)
    lenTest = float(lenTest)
    return numRight/lenTest

def compute_accuracy_Confusion(test_y, pred_y):
    lenTest = len(test_y)
    lenPred = len(pred_y)
    confusion_matrix = []
    pnay = 0
    pnan = 0
    pyay = 0
    pyan = 0
    numRight = 0
    numWrong = 0
    if(lenTest!=lenPred):
        return None
    for i in range(len(test_y)):
        if(test_y[i] > 0 and pred_y[i] > 0):
            pyay += 1
            numRight += 1
        elif(test_y[i] > 0 and pred_y[i] < 0):
            pnay += 1
            numWrong += 1
        elif(test_y[i] < 0 and pred_y[i] > 0):
            pyan += 1
            numWrong +=1
        else:
            pnan += 1
            numRight +=1
    confusion_matrix.append( [pnan, pyan])
    confusion_matrix.append( [pnay, pyay])
    numRight = float(numRight)
    lenTest = float(lenTest)
    return confusion_matrix

def get_id():
    return 'tuc48831'

def createTrain_xAndy(training_data, num_train):
    train_x=[]
    train_y=[]
    #only get as many examples as I ask for
    for i in range(num_train):
        #generate a random number and get the data from the training set
        ran = np.random.randint(0,15000)
        holder = training_data[ran]
        #slice off the target variable and append it to the array for the x, keep only the target for the y
        train_x.append(holder[1:])
        train_y.append(holder[:1])
    #turn it into a numpy array and return it
    train_x = np.array(train_x, np.float64)
    train_y = np.array(train_y, np.str)
    return train_x, train_y

def createTest_xAndy(testing_data):
    test_x=[]
    test_y=[]
    #capture the whole testing set
    for item in testing_data:
        holder = item
        #slice off the target variable and append it to the array for the x, keep only the target for the y
        test_x.append(holder[1:])
        test_y.append(holder[:1])
    #turn it into a numpy array and return it
    test_x = np.array(test_x, np.float64)
    test_y = np.array(test_y, np.str)
    return test_x, test_y

def main():
    
    train_x, test_x, train_y, test_y = getData()
    
   
    #list of the different sizes i need to train
    num_train = [100, 1000, 2000, 5000, 10000, 15000]
    confusion = True
    confusion_table = []
    #loop through the different sizes of training sets
    for i in num_train:
        #run pla on every letter, this is OVA
        #mark the time
        start = time.time()
        rollingAccuracy = 0.0
        for letter in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']:
            #generate sets I need
            train_x, train_y = createTrain_xAndy(training_data, i)
            test_x, test_y = createTest_xAndy(testing_data)
            
            #create a set of training data with a bias term, adjust y to be +/- 1
            temp_x = adjustX(train_x)
            temp_y = adjustY(train_y, letter)
            
            #i settled on 2 * sqrt of the size of the training set kind of arbitrarily
            #generate w
            w = train_pocket(temp_x, temp_y, 2*np.sqrt(i))
            #i need to adjust x
            temp_x = adjustX(test_x)
            #get my predictions for y
            pred_y = test_pocket(w, temp_x)
            #adjust my predicitions so that it is +/-1 instead of 'a', 'b', etc.
            temp_y = adjustY(test_y, letter)
            #get the accuracy
            acc = compute_accuracy(temp_y, pred_y)
            #add it to my total accuracy
            rollingAccuracy += acc
            if(confusion):
                confusion_table = compute_accuracy_Confusion(temp_y, pred_y)
                confusion = False
        end = time.time()
        duration = end-start
        rollingAccuracy /= 26
        #mark the end time, get the duration, get the average over the whole alphabet, print the information
        print("For pla with num_train: " + str(i) + 
              " averaged over all 26 letters is: " + str(rollingAccuracy) + 
              " the time to execute is: " + str(duration))
        
        #numpy arrays are referential, so any edits I do happen in the real object, not a copy of the object in a functions stack
        #this means i need to regenerate the training set to "undo" the edits i needed for OVA
        train_x, train_y = createTrain_xAndy(training_data, i)
        test_x, test_y = createTest_xAndy(testing_data)
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
        train_x, train_y = createTrain_xAndy(training_data, i)
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
 
    
    print(confusion_table)
    
    #acc = compute_accuracy(test_y, pred_y)
    
    id = get_id()

if __name__ == "__main__":
    main()
