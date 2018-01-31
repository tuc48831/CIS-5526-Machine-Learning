"""
Spyder Editor

This is a temporary script file.
"""

###IMPORTANT NOTE: I am not sure I am using the RNN stuff from tensorflow properly, I feel like I'm not capturing the time series part of the data properly
#my plan of action over thanksgiving break is to try to purposefully overfit it like you mentioned in the lecture yesterday 11.16
#my code currently runs and can generate plots of predictions vs real values, but it is far from ideal
#I also have yet to figure out a normalization scheme, this would help my losses be more clear since my numbers are all tiny floats close to 0 and thus the losses are as well
#I will add more layers and check other things after I get it properly overfitted

import numpy as np
from sklearn import tree
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import graphviz

#a function to get my data
##IMPLEMENT MORE FEATURE ENGINEERING: for now I did some small changes directly to the data and have 2 datasets, but I haven't yet programmed
##any of my feature engineering, I will focus on that once I feel good about my RNN implementation being overfit
def getData(indicator):
    filename = "rawData.txt"
    with open(filename) as file:
        #skip the headers
        next(file)
        #initialize the holding arrays
        lines = []
        total_x = []
        total_y = []
        #go line by line, removing extraneous info, splitting into an array, append it to total lines
        for line in file:
            line = line.strip("\n")
            line = line.split(",")
            lines.append(line[1:])
        
        lines = np.array(lines)
        lines = lines.astype(np.float)
        ###if indicator == 1 FEATURE ENGINEERING on x data
        if(indicator == 1):        
            lines = expandData(lines)
        
        for i in range(len(lines) - 2):
            opening_price = float(lines[i][0])
            target_price = float(lines[(i+1)][3])
            #skip the date colume
            line_x = lines[i]
            #zero centering the x data and converting it to percentages
            #start at one to skip over the opening price column as it will just be 0 after subtracting 1
            for j in range(1, len(line_x)):
                #don't adjust the volume column
                if(j != 5):
                    #make it a percent
                    line_x[j] = float(float(line_x[j]) / opening_price)
                    #subtract 1
                    line_x[j] = float(line_x[j]) - 1
            #get the last item as the target variable
            line_y = float(target_price / opening_price) - 1
            #zero center the y
            line_y = float(line_y)
            
            #convert to np array
            line_x = np.array(line_x)
            line_y = np.array(line_y)
            #convert to a float array instead of string
            line_x = line_x.astype(np.float)
            line_y = line_y.astype(np.float)
            #cutting off the date since I won't need it
            total_x.append(line_x)
            total_y.append(line_y)
            

    #convert to numpy arrays, taking off the first 30 days because of my feature engineering max days was 30
    #and some of the lines wont have all the features
    total_x = np.array(total_x)
    total_y = np.array(total_y)
    
    print(total_x[0])
    #delete the volume column (index 5)
    total_x = np.delete(total_x, 5, 1)
    #delete the opening price column (index 0)
    total_x = np.delete(total_x, 0, 1)
    print(total_x[0])
    
    if(False):
        means = np.mean(total_x, axis = 0)
        
        total_x = np.subtract(total_x, means)
        
        #std normalization on x
        std_devs = np.std(total_x, axis=0)
        maxes = np.amax(total_x, axis = 0)
        
        total_x = np.divide(total_x, maxes)
    else:
        pca = PCA(n_components=3, whiten=True)
        pca.fit(total_x)
        x_pca = pca.transform(total_x)
        
    print("original shape is: " + str(np.shape(total_x)))    
    print("new shape is: " + str(np.shape(x_pca)))
    
    print(total_x[0])
    print(x_pca[0])
    
    #chop off the days with missing averages
    total_x = x_pca[1440:]
    total_y = total_y[1440:]
    #some print checks to make sure everything is what I thought it should be
    print(total_x)
    print(total_y)
    print("the length of total_x is:" +str(len(total_x)))
    print("the length of total_x[0] is:" +str(len(total_x[0])))
    print("the length of total_y is:" +str(len(total_y)))
    print(total_x[0])
    print(total_x[0][0])
    print(total_x[0][1])
    #return a 2d np array with the first index being the row, and the 2nd index being the column
    return total_x, total_y, len(total_x[0]), len(total_x)

def expandData(total_x):
    #
    expandeded_x = total_x
    #one business week, 2 weeks, 1 month, 3 months, 6 months, 1 year
    days_to_do = [2, 3, 4, 5, 10, 22, 66, 132, 264]
    for i in days_to_do:
        print(i)
        expandeded_x = addMovingAverage(expandeded_x, i)
        expandeded_x = addVWAP(expandeded_x, i)
    return expandeded_x

def addMovingAverage(total_x, days):
    length = len(total_x)
    total_x = np.c_[total_x, np.zeros(length)]
    for i in range(days + 1, len(total_x)):
        #these will be averages of the closing price, and closing price's column index is 3
        subslice_of_days = total_x[i - days : i]
        total = 0
        for j in range(len(subslice_of_days)):
            total += subslice_of_days[j][3]
        total = float(total/days)
        total_x[i][-1] = total
    return total_x

def addVWAP(total_x, days):
    length = len(total_x)
    total_x = np.c_[total_x, np.zeros(length)]
    for i in range(days + 1, len(total_x)):
        #these will be averages of the closing price, and closing price's column index is 3
        subslice_of_days = total_x[i - days : i]
        sum_price_times_volume = 0
        sum_volume = 0
        for j in range(len(subslice_of_days)):
            #index of closing price is 3, and volume is 6
            sum_volume += subslice_of_days[j][5]
            sum_price_times_volume = subslice_of_days[j][3] * subslice_of_days[j][5]
        total = float(sum_price_times_volume)/float(sum_volume)
        total_x[i][-1] = total
    return total_x

def main(indicator):
    total_x, total_y, _ , _ = getData(indicator)
    clf = tree.DecisionTreeRegressor(max_depth=5)
    clf = clf.fit(total_x, total_y)
    
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("data")
    return

if __name__ == '__main__':
    main(0)
    #main(1)
    