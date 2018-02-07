# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
import random as rand

class PLA:
    
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.pred_y = None
        self.weights = None
    
    def calcErrorPLA(train_x, train_y, z, length):
        #from the calc Ein slide, get 1/n
        overN = 1/length
        #get the matrix multiplication of the transpose adn the weights
        geo = np.matmul(train_x, z)
        #subtract the y
        geo = geo - train_y
        #take the magnitude
        geo = np.linalg.norm(geo)
        #put it all together along with the squared element
        return overN*np.power(geo, 2)
    
    def train_pocket(self):
        #store the length, initialize w as my final and z as my temp
        length = len(self.train_x)
        #num iters is a np float 64 so i have to change it
        num_iters = int(2*np.sqrt(length))
        
        w = []
        #give z a non-random starting point by calculating the dot product of the pinv(x) and y
        w = np.dot(np.linalg.pinv(self.train_x), self.train_y)
        #try a random starting weight
        
        #then num iterations times
        for i in range(num_iters):
            #pick a random point
            test = rand.randint(0,length-1)
            #while the signs are not equal between wTx and y, keep repicking
            temp = length
            #this is super dangerous but I kept getting stuck looking for a misclassified point
            #if the data is somehow linearly seperable, this gets stuck in a loop
            #i got stuck in this loop 3 separate times so I invented an escape mechanism that is self-admittedly completely arbitrary
            while(np.dot(np.transpose(w), self.train_x[test]) * self.train_y[test] >= 0):
                temp-=1
                test = rand.randint(0,length-1)
                if(temp<0):
                    return w
            #calculate new weights
            temp = w + self.train_y[test]*self.train_x[test]
            #calculate errors of new weights and old weights
            errorTemp = self.calcErrorPLA(self.train_x, self.train_y, w, length)
            errorW = self.calcErrorPLA(self.train_x, self.train_y, w, length)
            #swap them if new weights result in less error
            if(errorTemp < errorW):
                w = temp
        w = np.array(w, np.float64)
        self.weights = w
        return None
    
    def test_pla(self):
        pred_y = []
        #w comes as a 17,1 array, and using squeeze is a view so it isn't kept between functions so I have to recall it here
        w = self.w.squeeze()
        #iterate through whole text_x
        for i in range(len(self.test_x)):
            #get dot product of weights and entry
            temp = np.dot(w, self.test_x[i])
            if(temp > 0):
                pred_y.append(1)
            elif(temp < 0):
                pred_y.append(-1)
            else:
                pred_y.append(0)
        pred_y = np.array(pred_y)
        self.pred_y = pred_y
        return pred_y
    
def testing():
    return None

def main():
    return None

if __name__ == "__main__":
    testing()