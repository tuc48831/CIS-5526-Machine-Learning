# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
import random as rand

class PLA:
    
    #tested and seems ok
    def __init__(self, train_x, train_y, test_x):
        self.train_x = train_x
        self.num_examples = len(train_x)
        self.x_length = len(train_x[0])
        self.train_y = train_y
        self.test_x = test_x
        self.pred_y = None
        self.weights = None
    
    #tested and seems ok
    ###A giant side note, using the pseudo inverse to solve the least squares problem means if you use squared error the weights will never update
    ###Also, squared error is definitely not the best, when run on the sample data set, square error gets ~40% correct, but percent errors gets about ~95% correct
    def calc_square_error_PLA(self):
        ###Changed this function from a fancy way to a simpler way, while it might take more time i find it more human readable
        #create total error variable
        total = 0.0
        #iterate through entire training set
        for i in range(self.num_examples):
            #take w transpose and multiply it to the current example
            temp = np.dot(self.weights, self.train_x[i])
            #get the squared error by subtracting y and squaring it
            temp = temp - self.train_y[i]
            temp = temp**2
            #add it to the total
            total += temp
        #get the average error by dividing the total error by the number of examples
        total = total / self.num_examples
        #return the total error
        return total
    
    #tested and works
    def calc_percent_error_PLA(self):
        ###Changed this function from a fancy way to a simpler way, while it might take more time i find it more human readable
        #create total error variable
        counter = 0
        #iterate through entire training set
        for i in range(self.num_examples):
            #take w transpose and multiply it to the current example
            temp = np.dot(self.weights, self.train_x[i])
            #get the squared error by subtracting y and squaring it
            if(temp > 0 and self.train_y[i] > 0):
                counter += 1
            elif(temp <= 0 and self.train_y[i] <= 0):
                counter += 1
        #get the average error by dividing the total error by the number of examples
        total = 1 - (counter / self.num_examples)
        #return the total error
        return total
    
    def train_pocket(self):
        #num iters is a np float 64 so i have to change it
        num_iters = int(20*np.sqrt(self.num_examples))
        #give z a non-random starting point by calculating the dot product of the pinv(x) and y
        self.weights = np.dot(np.linalg.pinv(self.train_x), self.train_y)
        #print(np.shape(self.weights))
        #print(self.weights)
        #calculate the error for the initial starting weights
        errorW = self.calc_percent_error_PLA()
        #print("The starting error is: " + str(errorW))
        
        #then num iterations times
        for i in range(num_iters):
            #pick a random point
            test = rand.randint(0, self.num_examples-1)
            #this counter allows escaping the below loop, if the data is linearly seperable, it could get stuck looking for a misclassified point
            counter = self.num_examples
            #while the signs are equal between wTx and y, keep repicking
            while( (np.dot(self.weights, self.train_x[test]) > 0 and self.train_y[test] > 0) 
                or (np.dot(self.weights, self.train_x[test]) <= 0 and self.train_y[test] <= 0)
                and counter > 0):
                #print("test is: " + str(test))
                #print(np.dot(self.weights, self.train_x[test]))
                test = rand.randint(0, self.num_examples-1)
                counter-=1
            #hold the previous weights
            holder = self.weights
            #calculate new weights
            self.weights = self.weights + np.multiply((self.train_y[test] - np.dot(self.weights, self.train_x[test])), self.train_x[test])
            #calculate errors of new weights and old weights
            errorTemp = self.calc_percent_error_PLA()
            #print("the temp error is: " + str(errorTemp))
            #swap them if new weights result in less error
            if(errorTemp < errorW):
                #print("actually updated the weights")
                errorW = errorTemp
            #else, reset the weights
            else:
                self.weights = holder
        self.weights = np.array(self.weights, np.float64)
        return None
    
    def generate_predictions(self):
        pred_y = []
        #w comes as a 17,1 array, and using squeeze is a view so it isn't kept between functions so I have to recall it here
        w = self.weights.squeeze()
        #iterate through whole text_x
        for i in range(len(self.test_x)):
            #get dot product of weights and entry
            temp = np.dot(w, self.test_x[i])
            if(temp > 0):
                pred_y.append(1)
            elif(temp <= 0):
                pred_y.append(0)
            else:
                pred_y.append(0)
        pred_y = np.array(pred_y)
        self.pred_y = pred_y
        return pred_y
    
def testing():
    #some dummy data for my function testing
    #for x<=5 the target variable is 1, and where x > 5 it is 0
    #with a bias term of 1 for all training points
    train_x = [[1,1,1],[1,2,2],[1,3,3],[1,4,4],[1,5,5],[1,6,6],[1,7,7],[1,8,8],[1,9,9],[1,10,10]]
    train_y = [1,1,1,1,1,0,0,0,0,0]
    #the test set is  
    test_x = [[1,2,2],[1,3,3],[1,7,7],[1,9,9]]
    
    #testing the creation of a PLA object
    obj = PLA(train_x, train_y, test_x)
    print("The object variable train_x is: " + str(obj.train_x))
    print("The object variable num_examples is: " + str(obj.num_examples))
    print("The object variable x_length is: " + str(obj.x_length))
    print("The object variable train_y is: " + str(obj.train_y))
    print("The object variable test_x is: " + str(obj.test_x))
    print("The object variable pred_y is: " + str(obj.pred_y))
    print("The object variable weights is: " + str(obj.weights))
    
    #testing the train_pocket function
    print("TESTING THE train_pocket FUNCTION:")
    obj.train_pocket()
    print("after train_pocket the variable 'obj.weights' is: " + str(obj.weights))
    
    #testing the calc_square_error_pla function
    print("TESTING THE calc_square_error_pla FUNCTION:")
    print("the error is: " + str(obj.calc_square_error_PLA()))
    
    #testing the calc_percent_error_pla function
    print("TESTING THE calc_percent_error_pla FUNCTION:")
    print("the error is: " + str(obj.calc_percent_error_PLA()))
    
    #testing the generate_predictions function
    print("TESTING THE generate_predictions FUNCTION:")
    print("the predictions are: " + str(obj.generate_predictions()))
    
    return None

def main(self):
    self.train_pocket()
    self.generate_predictions()
    return self.pred_y

if __name__ == "__main__":
    testing()