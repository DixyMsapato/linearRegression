import math
import numpy
import csv
from matplotlib import pyplot as plt
import os


#Assuming the data is all integer data
#And of a particular format, the getTrainingData needs modifications
#Depending on number of features of data
#We convert the data from getTrainingData from string format to

def getTrainingData():
    #call input function
    filepath = input('Please enter the file path for your data: ')
    if os.path.exists(filepath):
        with open(filepath, 'r') as trainingData:
            csv_reader = csv.reader(trainingData, delimiter = ',')
            header = next(csv_reader)
            #load the data into a list of lists
            data = [row for row in csv_reader]
            colNum = len(data[0])

            #trueColNum ignores the columns with non-numerical data
            #The 3 is comes of our particular example data set
            #of which the last 3 columns are non-numerical/not important

            trueColNum = colNum

            #debugging print statements
            #print(colNum)
            #print(data[0])

           #This for loop changes the data entries from string to int

            intData = []

            for line in data:
                intLine = []
                print(line)
                counter = 0
                # while loop which converts line into a line of float instead of str values
                # we want float not into so that when we normalise, division by
                # standard deviation won't result in zero entries
                while counter < trueColNum:
                    intLine.append(float(line[counter]))
                    counter = counter + 1

                intData.append(intLine)
            # This turns 2d array into a numpy 2d array i.e matrix
            # Thus can now use numpy operations on our data!
            return numpy.array(intData)
    elif filepath == 'quit':
        return
    else:
        print("The path you entered does not exists, please try again or type quit to exit the program")
        #Call getTrainingData() again.
        getTrainingData()

trainingData = getTrainingData()
#This turns 2d array into a numpy 2d array i.e matrix
#Thus can now use numpy operations on our data!

#We first split the data into training variables X and output y
#the output is in the first column of the trainingData
y = trainingData[:,[0]]
#create ThetaZero column on int 1s
thetaZero = numpy.ones((trainingData.shape[0],1),int)
#Remove the y values from training data
Xprime = numpy.delete(trainingData,0,1)
#The design matrix X
X = numpy.append(thetaZero, Xprime,1)

def featureNormalise(Xnorm):
    mean = numpy.mean(Xnorm,0)
    standDev = numpy.std(Xnorm,0)
    colNum = Xnorm.shape[1]
    #while loop starts at 1 because dont want to apply normalisation to
    #the x_0 values corresponding to theta zero
    i = 1
    rowNum = Xnorm.shape[0]

    while i < colNum:
        mu = mean[i]*(numpy.ones((1,rowNum)))
        stand = standDev[i]*(numpy.ones((1,rowNum)))
        Xnorm[:,i] = numpy.true_divide((Xnorm[:,i]-mu),stand)
        i = i +1

    return Xnorm


#compute the cost fucntion for linear regression with multivariable(features)
#data: training data X is an mxn array, m =num training examples, n= num features
#y is an mx1 vector with the y values for the training examples
#theta is an nx1 vector
def costFunction(X,y,theta):
    #m = number of training examples i.e length(y)
    #add code checking dimensions MATCH ***HERE****
    m = y.shape[0]
    J = (1 / (2 * m)) * numpy.sum((numpy.square(X.dot(theta) - y)));
    return J

#Normalise data


y = trainingData[:,[0]]
theta = numpy.zeros((X.shape[1],1))
alpha = 0.01
print(costFunction(X,y,theta))


def gradientDescent(X,y,theta,alpha,num_iters):
    m = X.shape[0]
    numFeatures = X.shape[1]
    iters = 0
    J_history = numpy.zeros((num_iters, 1))
    gradJ = numpy.zeros((numFeatures,1))
    temp = []
    for iters in range(0,num_iters):

        for x in range(0,numFeatures):
            temp = (X.dot(theta)-y)*(X[:,x].reshape(m,1))

            gradJ[x] = numpy.sum(temp)
        #print(gradJ)
        theta = theta - ((alpha / m) * gradJ)
        print('Iteration number:',iters)
        J_history[iters] = costFunction(X, y, theta)

    print('The cost function history',J_history)
    print('Final values of theta', theta)

    return theta
t = gradientDescent(X,y,theta,alpha,1500)

Xplot = numpy.delete(X,0,1)
print(Xplot.shape)
print(y.shape)
#Scatter plot of original data
plt.scatter(Xplot,y)
#Plot of hypothethis line of best fit
plt.plot(Xplot, X.dot(t))
plt.show()