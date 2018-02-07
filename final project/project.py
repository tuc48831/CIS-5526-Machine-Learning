# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

###IMPORTANT NOTE: I am not sure I am using the RNN stuff from tensorflow properly, I feel like I'm not capturing the time series part of the data properly
#my plan of action over thanksgiving break is to try to purposefully overfit it like you mentioned in the lecture yesterday 11.16
#my code currently runs and can generate plots of predictions vs real values, but it is far from ideal
#I also have yet to figure out a normalization scheme, this would help my losses be more clear since my numbers are all tiny floats close to 0 and thus the losses are as well
#I will add more layers and check other things after I get it properly overfitted

import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from statistics import mean

#a function to get my data
##IMPLEMENT MORE FEATURE ENGINEERING: for now I did some small changes directly to the data and have 2 datasets, but I haven't yet programmed
##any of my feature engineering, I will focus on that once I feel good about my RNN implementation being overfit
def getData(indicator, component_num):
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
        keep_length = len(lines[0]-2)
        ###if indicator == 1 FEATURE ENGINEERING on x data
        if(indicator == 1):        
            lines = expandData(lines)
        print(lines[0])
        
        ###covariance and correlation stuff here
        print("The covariance matrix is: ")
        example = np.cov(lines, rowvar=False)
        print(example)
        #for each feature column
        for i in range(len(lines[0] - 1)):
            column = lines[:, i]
            target_var = lines[:, 3]
            print( str(i) + "th column's correlation with the target var (3rd column) is:")
            print(np.corrcoef(column, target_var))
        
        
        for i in range(len(lines) - 2):
            opening_price = float(lines[i][0])
            target_price = float(lines[(i+1)][3])
            #skip the date colume
            line_x = lines[i]
            #zero centering the x data and converting it to percentages
            #start at one to skip over the opening price column as it will just be 0 after subtracting 1
            for j in range(0, len(line_x)):
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
    
    #delete the volume column (index 5), it was completely useless when i made the decision trees
    total_x = np.delete(total_x, 4, 1)
    #delete the opening price column (index 0)
    total_x = np.delete(total_x, 0, 1)
    print(total_x[0])
    
    #split into training and testing data
    train_x, test_x, train_y, test_y = generateTrainTest(total_x, total_y, 0.8)
    
    #normalize the data by centering it and max scaling it
    means = np.mean(train_x, axis = 0)
    train_x = np.subtract(train_x, means)
    #apply the training transformations to the test x data
    test_x = np.subtract(test_x, means)
    #std normalization on x
    std_devs = np.std(train_x, axis=0)
    maxes = np.amax(train_x, axis = 0)
    #apply the training transformations to the test x data
    train_x = np.divide(train_x, maxes)
    
    #recalc the covariance matrix
    print("The covariance matrix before PCA is: ")
    example = np.cov(total_x, rowvar=False)
    print(example)
    
    #run a PCA analysis to reduce the high covariances
    components = len(total_x[0])
    if(indicator == 1):
        components = component_num
    pca = PCA(n_components=components, whiten=True)
    pca.fit(total_x)
    #get the new training x from the pca
    train_x = pca.transform(train_x)
    #apply the training transformations to the test x data
    test_x = pca.transform(test_x)
    
    #sanity check print statemnets
    print("The covariance matrix after PCA is: ")
    example = np.cov(total_x, rowvar=False)
    print(example)
    for i in range(len(total_x[0])):
            column = total_x[:, i]
            print( str(i) + "th column's correlation with the target var after PCA is:")
            print(np.corrcoef(column, total_y))
    
    print("original shape is: " + str(np.shape(total_x)))    
    
    
    #chop off the days with missing averages
    train_x = train_x[264:]
    train_y = train_y[264:]
    test_x = test_x[264:]
    test_y = test_y[264:]
    #some print checks to make sure everything is what I thought it should be
    print(total_x)
    print(total_y)
    print("the length of trainl_x is:" +str(len(train_x)))
    print("the length of train_x[0] is:" +str(len(train_x[0])))
    print("the length of train_y is:" +str(len(train_y)))
    #return a 2d np array with the first index being the row, and the 2nd index being the column
    return train_x, train_y, test_x, test_y, len(train_x[0]), len(total_x)

def expandData(total_x):
    #a simple reassignment for basically no reason
    expanded_x = total_x
    #all the days under 1 week inclusive, 2 weeks, 1 month, 3 months, 6 months, 1 year (all in business days)
    days_to_do = [2, 3, 4, 5, 10, 22, 66, 132, 264]
    #for each day
    for i in days_to_do:
        print(i)
        #add the moving average, the vwap, and the volatility to the x data
        expanded_x = addMovingAverage(expanded_x, i)
        expanded_x = addVWAP(expanded_x, i)
        expanded_x = addStdDev(expanded_x, i)
    #return the new expanded x
    return expanded_x

#this function adds a moving average to the x data of the past days input
def addMovingAverage(total_x, days):
    #get the length
    length = len(total_x)
    #add a column of zeroes to the end
    total_x = np.c_[total_x, np.zeros(length)]
    #for all the days its possible
    for i in range(days + 1, length):
        #these will be averages of the closing price, and closing price's column index is 3
        subslice_of_days = total_x[i - days : i]
        total = 0
        for j in range(days):
            total += subslice_of_days[j][3]
        #calculate the average by dividing the total by the number of days
        total = float(float(total)/float(days))
        total_x[i][-1] = total
    return total_x

#this function adds a volume weighted average price to the x data of the past days input
def addVWAP(total_x, days):
    #get the length
    length = len(total_x)
    #make a column of all zeroes at the end
    total_x = np.c_[total_x, np.zeros(length)]
    #start at days_1 and calculate for the rest of the days
    #this formula can be understood better here: https://www.investopedia.com/terms/v/vwap.asp
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

#this function adds the volatility to the x data of the past days input
def addStdDev(total_x, days):
    #get the length
    length = len(total_x)
    #make a new column of all zeros at the end
    total_x = np.c_[total_x, np.zeros(length)]
    #start at days + 1 and calculate the number for all days
    for i in range(days + 1, length):
        #these will be averages of the closing price, and closing price's column index is 3
        subslice_of_days = total_x[i - days : i]
        #compute std devs along columns, get it for closing price
        total_x[i][-1] = np.std(subslice_of_days, axis=0)[3]
    return total_x

#a function to divide my total data into train and test sets
def generateTrainTest(total_x, total_y, split_ratio):
    #a simple uhoh catch to exit the whole program if I really messed up and the lengths of x and y dont match
    if(len(total_x) != len(total_y)):
        print("Length mismatch between X and Y, terminating")
        exit(-1)
    #capture the total length
    length = len(total_x)
    #get an index to split the data according to my split ratio
    split = int(length * split_ratio)
    #slice off train x, test x, train y, and test y
    train_x = total_x[:split]
    test_x = total_x[split:]
    train_y = total_y[:split]
    test_y = total_y[split:]
    #a print check to make sure it is what I think it should be
    print(type(train_x))
    #return my training and testing sets
    return train_x, test_x, train_y, test_y    

#a function to create a mini batch of data, generating a batch was mentioned on the tensorflow RNN tutorial
#but they're using dense embedded matrices for word prediction and not numbers so I had to think of a simple
#way to slice the array based off batch size
def makeBatch(train_x, train_y, batch_size):
    #calculate the number of batches by dividing the length of my training set by the batch size
    num_batches = int(len(train_x) / batch_size)
    #print(num_batches)
    
    #make a numpy array that is as long as my num batches
    indices = np.arange(num_batches)
    #shuffle the list
    np.random.shuffle(indices)
    #iterate through all of the batches
    for i in indices:
        #capture the slice of x by multiply the shuffled batch number times the batch size
        batch_x = train_x[i * batch_size : (i+1) * batch_size]
        #i have to expand this dimension or tensorflow throws an error
        #since it's a 32,10 array, and tensorflow expects a 32,10,1
        batch_x = np.expand_dims(batch_x, axis=2)
        #capture the slice of y by multiply the shuffled batch number times the batch size
        batch_y = train_y[i * batch_size : (i+1) * batch_size]
        #i have to expand this dimension otherwise tensorflow throws an error
        #since its a 32, array, and tensorflow expects a 32,1
        batch_y = np.expand_dims(batch_y, axis=1)
        #yield here allows me to keep this generator and access it continually for the next batch, ensuring that
        #every batch is process and no batch is repeated
        yield batch_x, batch_y

def visualize_loss(x, y, test_x, test_y, num_features, trainLength):
    list_learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    ret_vals = []
    for num in list_learning_rates:
        _, _, training_losses = trainRNN(0, x, y, test_x, test_y, num_features, trainLength, True, num)
        #run the training for x many epochs
        #get the loss from each epoch and append it to a list
        ret_vals.append(training_losses)
    #plot all the different things
    for i in range(len(list_learning_rates)):
        plt.plot(ret_vals[i], label=str(list_learning_rates[i]))
    plt.legend()
    plt.show()

#The entire point of this overfit function is to serve as a staging area for me to try and test different things to possibly incorporate in my trainRNN function
#you can pretty much ignore this function as it was only for testing/my benefit, but i don't want to delete it and make it seem like I didn't experiment with different things
def overfit(indicator):
    lstm_nodes_per_layer = 0
    num_layers = 2
    dont_drop_rate = 0.5
    starting_learning_rate = 0.01
    num_passes = 26
    batch_size = 32
    steps = 512
    decay_rate = 0.99
    #deprecated, data is no longer retrieved inside training function
    total_x, total_y, num_features, trainLength = getData(indicator)
    #an experiment into the number of nodes per layer
    lstm_nodes_per_layer = int(num_features * 4)
    
    print(np.shape(total_x))
    print(np.shape(total_y))
    total_x = np.resize(total_x, (40, num_features))
    total_y = np.resize(total_y, (40))
    trainLength = 120
    print(np.shape(total_x))
    print(np.shape(total_y))
    
    #split into training and testing data for both normal data and engineered
    train_x, test_x, train_y, test_y = generateTrainTest(total_x, total_y, split_ratio)
    print(len(train_x))
    #train_x_eng, test_x_eng, train_y_eng, test_y_eng = generateTrainTest(engineered_x, engineered_y, split_ratio)
    
    #create the tensorflow graph set it to the default
    #this is just making the things in the graph and not running the session
    graph1 = tf.Graph()
    with graph1.as_default():
        #a placeholder x for my input data when the session runs
        ##VERY concerned, I don't think this works how I think it should
        #dimensions will be (batch size, num features, num_features, 1)
        x = tf.placeholder(tf.float32, [None, num_features, 1])
        #a placeholder y for my target data when the session runs
        y = tf.placeholder(tf.float32, [None, 1])
        #my learning rate as a constant, i have seen some people do variable learning rates?
        global_step = tf.Variable(0, trainable=False)
        #learning rate decay (directly from tensorflow.org's page on learning rate decay)
        learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step,
                                                   steps, decay_rate, staircase=True)
        
        #initialize 1 layer of LSTM cells for my RNN, i would like to try more layers in the future
        cell = tf.contrib.rnn.LSTMCell(lstm_nodes_per_layer)
        
        #get the "return values" from the rnn operation on the input data
        val, state_ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        #transpose val because it is currently batch size, num stemps, nodes per layer
        #and make it become (num features, batch size, nodes per layer)
        val = tf.transpose(val, [1, 0, 2])
        #get the last set of weights 
        last = tf.gather(val, int(val.get_shape()[0]) - 1)
        # Define weights and biases
        # From your lecture on 11.16 I will look into doing something better instead of random, like Xavier initialization? I'm not sure how necessary for now since I'm just 1 layer
        weights = tf.get_variable("weights", shape=[lstm_nodes_per_layer, 1], initializer = tf.contrib.layers.xavier_initializer())
        #weights = tf.Variable(tf.random_normal([lstm_nodes_per_layer, 1]))
        #regularization?
        tf.contrib.layers.l2_regularizer(weights, scope=None)
        # I have no idea what good starting biases would be, random seemed like a safe guess
        bias = tf.Variable(tf.constant(0.1))
        # a prediction is wTx + b
        prediction = tf.matmul(last, weights) + bias
        
        
        #the cost function as reducing the mean squared error, just like you talked about in class yesterday, I'm not sure this is the best loss function
        #removed tf.abs(prediction - y)
        cost = tf.reduce_mean(tf.square((prediction - y) ))
        #This optimizer is probably not optimal (no pun intended), I want to look into different optimizers for regressions for RNNs
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
    
    preds = []
    ys = []
    

    
    #start training
    with tf.Session(graph=graph1) as sess:
        #with tf.device("/gpu:0"):
        #initialize everything
        tf.global_variables_initializer().run()
        
        #each step is looping through the data once
        for i in range(num_passes):
            #get a random batch from my makeBatch function and my training data
            for batch_x, batch_y in makeBatch(train_x, train_y, batch_size):
                #create a feed dictionary
                feed_dict = {x: batch_x, y: batch_y}
                #capture the loss from running the feed dictionary into my cost and optimizer
                train_loss, _ , train_pred, temp_y = sess.run([cost, optimizer, prediction, y], feed_dict)
                
                
                
            #print out incremental information to show the program is running and monitor the loss function
            if i % 5 == 0:
                #calculate batch accuracy & loss
                print("At step #" + str(i) + ", the minibatch loss is: " + str(train_loss))
                preds.append(train_pred)
                ys.append(temp_y)
        #print to know I'm done
        print("optimization finished")
        total = np.zeros((len(preds[0]), 3))
        for i in range(len(preds[0])):
            total[i][0] = preds[-1][i]*100
            total[i][1]  = ys[-1][i]*100
            total[i][2] = (preds[-1][i]-ys[-1][i])*100
        print(total)
        
    #left over plot showing code from before i made better graphs
    if(indicator == 1):
        #a quick scatter plot to see some of my predictions vs actuals to understand visually how off I am
        #ideally it would be a perfectly correlated line of y=x, but as of right now it looks like neither is even close
        for i in range(len(preds)):
            plt.scatter(preds[i], ys[i])
        plt.show()
        plt.close()
        plt.scatter(preds[-1], ys[-1])
        plt.show()
        plt.close
    else:
        for i in range(len(preds)):
            plt.scatter(preds[i], ys[i])
        plt.show()
        plt.close()
        plt.clf()
        plt.scatter(preds[-1], ys[-1])
        plt.show()
        plt.close

def trainRNN(indicator, train_x, train_y, test_x, test_y, num_features, train_length, optional_learn, input_learn_rate):
#THESE PARAMETERS ARE MOSTLY PLACEHOLDERS
#I'm not exactly sure of the best RNN to set up so right now
#I just have these values, I will do testing/studying on them to figure out how I want my RNN to behave in the future
    #number of nodes per layer, this is 2 times the number of features, a figure i read listed as a safe upper limit in a previous research paper about financial regression
    lstm_nodes_per_layer = 8
    #num layers remained 1, i experimented with more layers but they increased training time while not improving accuracy
    num_layers = 1
    #dropout rate
    dont_drop_rate = 0.8
    #this is a starting learning rate that is subject to decay as more data is seen hopefully as a way to prevent overfitting
    starting_learning_rate = 0.001
    if(optional_learn):
        starting_learning_rate = input_learn_rate
    #total number of passes through training data
    num_passes = 5
    if(optional_learn):
        #to visualize the loss the number of passes is lessened to speed graph creation
        num_passes = 3
    #batch size is a power of 2
    batch_size = 32
    #num steps for each iteration of learning rate decay
    steps = 12800
    #decay rate
    decay_rate = 0.96
    
    #create the tensorflow graph set it to the default
    #this is just making the things in the graph and not running the session
    graph = tf.Graph()
    with graph.as_default():
        #a placeholder x for my input data when the session runs
        #dimensions will be (batch size, num features, num_features, 1)
        x = tf.placeholder(tf.float32, [None, num_features, 1])
        #a placeholder y for my target data when the session runs
        y = tf.placeholder(tf.float32, [None, 1])
        #a global step variable to help decay my learning rate
        global_step = tf.Variable(0, trainable=False)
        #learning rate decay (directly from tensorflow.org's page on learning rate decay)
        learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step,
                                                   steps, decay_rate, staircase=True)
        #initialize 1 layer of LSTM cells for my RNN
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(lstm_nodes_per_layer, layer_norm=True, activation=tf.nn.relu, dropout_keep_prob=dont_drop_rate)
        
        #get the "return values" from the rnn operation on the input data
        val, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        #transpose val because it is currently batch size, num stemps, nodes per layer
        #and make it become (num features, batch size, nodes per layer)
        val = tf.transpose(val, [1, 0, 2])
        #get the last set of weights
        last = tf.gather(val, int(val.get_shape()[0]) - 1)
        # Define weights and biases
        # From your lecture on 11.16 I will look into doing something better instead of random, like Xavier initialization? I'm not sure how necessary for now since I'm just 1 layer
        #weights = tf.get_variable("weights", shape=[lstm_nodes_per_layer, 1], initializer = tf.contrib.layers.xavier_initializer())
        #i tried xavier initialization and it did not seem to do good things to my data
        weights = tf.Variable(tf.multiply(tf.random_normal([lstm_nodes_per_layer, 1]), tf.sqrt(tf.divide(tf.cast(2.0, dtype=tf.float32), tf.cast(lstm_nodes_per_layer, dtype=tf.float32)))))
        #some L2 regularization
        tf.contrib.layers.l2_regularizer(weights, scope=None)
        #random biases
        bias = tf.Variable(tf.random_normal([1]))
        # a prediction is wTx + b
        prediction = tf.matmul(last, weights) + bias
        #the cost function as reducing the mean squared error, just like you talked about in class yesterday, I'm not sure this is the best loss function
        cost = tf.reduce_mean(tf.square(prediction - y))
        #This optimizer is RMS Prop as it also helps to reduces the learning rate based off of recent gradients
        #i ran into a lot of problems where the loss would jump upwards dramatically in the middle of the problem and this seemed to help with that
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
    
    #holding arrays for various things that will be retrieved from the network as it trains and runs
    preds = []
    ys = []
    training_losses = []
    
    #start training
    with tf.Session(graph=graph) as sess:
        #with tf.device("/gpu:0"):
        #initialize everything
        tf.global_variables_initializer().run()
        train_loss = 0
        #each pass is looping through the data once
        for i in range(num_passes):
            #get a random batch from my makeBatch function and my training data
            for batch_x, batch_y in makeBatch(train_x, train_y, batch_size):
                #create a feed dictionary
                feed_dict = {x: batch_x, y: batch_y}
                #capture the loss from running the feed dictionary into my cost and optimizer
                train_loss, _ , train_pred, temp_y = sess.run([cost, optimizer, prediction, y], feed_dict)
                #append the recent training losses to a record keeping array
                training_losses.append(train_loss)
            #print out incremental information to show the program is running and monitor the loss function
            if i % 2 == 0:
                #calculate batch accuracy & loss
                print("At step #" + str(i) + ", the minibatch loss is: " + str(train_loss))
                preds.append(train_pred)
                ys.append(temp_y)
        #print to know I'm done
        print("optimization finished, final loss is: " + str(train_loss))
        #print(preds[-1])
        #print(ys[-1])
        #print((preds[-1] - ys[-1]))
    
        #run and get predictions on the test X data
        
        #expand the dimensions of the test data like i did in the batch function to make it compatible
        test_x = np.expand_dims(test_x, axis=2)
        #create a feed duct of just the test x data
        feed_dict = {x: test_x}
        #get the predictions by running it through the network
        predictions = sess.run([prediction], feed_dict)
        
        #print("shape of test_y is: " + str(np.shape(test_y)))
        #test_y = np.expand_dims(test_y, axis=1)
        
    #below are some sanity checks of print statements as well as some important capturing of the MSE and some predictions/test variables to graph
        print("shape of predictions is: " + str(np.shape(predictions)))
        print("shape of test_y is: " + str(np.shape(test_y)))
        test_y = np.expand_dims(test_y, axis=1)
        
        mse = mean_squared_error(test_y, predictions[0])
        print("the mse is: " + str(mse))
        
        predictions = np.array(predictions)[0, -100:]
        test_y = np.array(test_y)[-100:]
        
        
        print("shape of predictions is: " + str(np.shape(predictions)))
        print("shape of test_y is: " + str(np.shape(test_y)))
        
        plt.plot(predictions, color="red", label="predictions")
        plt.plot(test_y, color="blue", label="actual y's")
        plt.legend()
        plt.show()
        plt.close()
        
        diffs = np.subtract(test_y, predictions)
        #plt.plot(diffs, color="green")
        #plt.show()
        
        
        return diffs, mse, training_losses

if __name__ == '__main__':
#these 2 overfits are left over from my experimentation
    #overfit(0)
    #overfit(1)
    
    #get total of data from the file, the engineering happens in the get data function
    train_x, train_y, test_x, test_y, num_features, trainLength = getData(0, 0)
    
    #get total of data from the file, the engineering happens in the get data function
    modtrain_x, modtrain_y, modified_test_x, modified_test_y, modified_num_features, modified_trainLength = getData(1, num_features)
    
    print("Sanity check")
    #this sanity check shows that after the PCA of my different data sets, their first elements are not the same so the 2 training x's are different
    print(train_x[0])
    print(modtrain_x[0])
    
    #a graph to visualize the loss curves of the different learning rates
    visualize_loss(train_x, train_y, test_x, test_y, num_features, trainLength)
    #sys.exit()
    mse0list = []
    mse1list = []
    #run the experiment 3 times
    for i in range(3):
        #get the MSE and the differences b/wn the predictions and the actual y's from running the network on normal data
        diffs0, mse0, _ = trainRNN(0, train_x, train_y, test_x, test_y, num_features, trainLength, False, 0)
        #get the MSE and the differences b/wn the predictions and the actual y's from running the network on data with extra features
        diffs1, mse1, _ = trainRNN(1, modtrain_x, modtrain_y, modified_test_x, modified_test_y, modified_num_features, modified_trainLength, False, 0)
        
        plt.plot(diffs0, color="blue", label="normal data")
        plt.plot(diffs1, color="red", label="domain engineered data")
        plt.legend()
        plt.show()
        
        print("mse0 is: " + str(mse0))
        mse0list.append(mse0)
        print("mse1 is: " + str(mse1))
        mse1list.append(mse1)
        
    #print the lists of the MSE
    print("The Mean Square Errors of the normal data is: ")
    print(mse0list)
    print("The Mean Square Errors of the domain engineered data is: ")
    print(mse1list)
    #show the averages
    print("The average mse of the normal data is: " + str(mean(mse0list)))
    print("The average mse of the domain engineered data is: " + str(mean(mse1list)))