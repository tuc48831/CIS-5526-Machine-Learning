# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance as dist

class kNN:
    
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.condense_idx = None
        self.pred = None
    
    def findNNs(self, num_nn):
    
        #initialize arrays
        neighbors = []
        #use cdist to calculate all distances at once
        temp = []    
        temp = dist.cdist(self.test_x, self.train_x, 'euclidean')
        #sort the array by the distances
        for i in range(len(temp)):
            #print(temp[i])
            #dont include the 0th element, it will be 0 because the lowest distance is to itself
            index_y = temp[i].argsort()[1:num_nn+1]
            instance = []
            
            for y in index_y:
                instance.append(self.train_y.item(y))
            neighbors.append(instance)
        #make it a numpy array and return it
        #print(str(neighbors))
        return neighbors
    
    def test_knn(self, num_nn):
        pred_y = []
        #iterate through test data
        #for i in range(len(test_x)):
        
        #find the array corresponding to the num NN closest neighbors
        tempClosest = self.findNNs(self.test_x, self.train_x, self.train_y, num_nn)
        #run some statistics on that array
        
        for i in range(len(tempClosest)):
            string_counter = {}
            for k in tempClosest[i]:
                if k in string_counter:
                    string_counter[k] += 1
                else:
                    string_counter[k] = 1
            sortedList = sorted(string_counter, key = string_counter.get, reverse = True)
            prediction = sortedList[0]
            #append that prediction to my return list
            pred_y.append(prediction)
        
        
        pred_y = np.array(pred_y)
        return pred_y
    
    def condense_data(self):
        condensed_idx = []
        ran = np.random.randint(0,len(self.train_x))
        
        #initialize subset with a single training example
        condensed_idx.append(ran)
        #create an array structure to house data and classify every point as being neighbored to the original one
        allTrain = []
        for i in range(len(self.train_x)):
            allTrain.append(condensed_idx[0])
        #iterate through the list and check if my actual classification differs from what I've been assigned
        correctly_classified = [];
        incorrectly_classified = [];
        for i in range(len(allTrain)):
            if self.train_y.item(allTrain[i]).lower() == self.train_y.item(i).lower():
                correctly_classified.append(i)
            else:
                incorrectly_classified.append(i)
        temp = 0;       
        while( len(incorrectly_classified) > 0  and len(incorrectly_classified) > temp):
            ran = np.random.randint(0,len(incorrectly_classified))
            condensed_idx.append(incorrectly_classified[ran])
            new_c_c = []
            new_i_c = []
            
            for i in range(len(incorrectly_classified)):
                incorrectly_classified[i] = incorrectly_classified[ran]
            
            for i in range(len(incorrectly_classified)):
                
                if self.train_y.item(allTrain[incorrectly_classified[ran]]).lower() == self.train_y.item(i).lower():
                    new_c_c.append(i)
                else:
                    new_i_c.append(i)
            temp = len(incorrectly_classified)
            correctly_classified = new_c_c
            incorrectly_classified = new_i_c
            
            #for an incorrectly classi
        condensed_idx = np.array(condensed_idx)
        self.condensed_idx = condensed_idx
        return condensed_idx
    
    def condenseTraining(self):
        temp_x = []
        temp_y = []
        for i in range(len(self.condensed_idx)):
            temp_x.append(self.train_x[i])
            temp_y.append(self.train_y[i])
        temp_x = np.array(temp_x)
        temp_y = np.array(temp_y)
        return temp_x, temp_y