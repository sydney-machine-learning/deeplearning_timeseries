#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from numpy import array
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame


# In[5]:


def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# In[6]:


def main():

    for i in range(1, 8) : 

        problem = i
        if problem ==1:
            traindata = np.loadtxt("Lazer/train.txt")
            testdata= np.loadtxt("Lazer/test.txt")
            name= "Lazer"
        if problem ==2:
            traindata = np.loadtxt(  "Sunspot/train.txt")
            testdata= np.loadtxt( "Sunspot/test.txt")
            name= "Sunspot"
        if problem ==3:
            traindata = np.loadtxt("Mackey/train.txt")
            testdata= np.loadtxt("Mackey/test.txt")  
            name= "Mackey"
        if problem ==4:
            traindata = np.loadtxt("Lorenz/train.txt")
            testdata= np.loadtxt("Lorenz/test.txt")  
            name= "Lorenz"
        if problem ==5:
            traindata = np.loadtxt( "Rossler/train.txt")
            testdata= np.loadtxt( "Rossler/test.txt")
            name= "Rossler"
        if problem ==6:
            traindata = np.loadtxt("Henon/train.txt")
            testdata= np.loadtxt("Henon/test.txt")
            name= "Henon"
        if problem ==7:
            traindata = np.loadtxt("ACFinance/train.txt") 
            testdata= np.loadtxt("ACFinance/test.txt")
            name= "ACFinance"
        
        print(name)
        print(traindata)
        print(testdata)
        
        train = traindata[0,0:3]
        for x in traindata:
            train = np.concatenate((train, x[3:5]))
        test = testdata[0,0:3]
        for x in testdata:
            test = np.concatenate((test, x[3:5]))


        # choose a number of time steps
        n_steps_in, n_steps_out = 5,10

        # split into samples
        train_X, train_Y = split_sequence(train, n_steps_in, n_steps_out)

        train = np.concatenate((train_X, train_Y), axis=1)
        train = pd.DataFrame(train, columns = ['Input1', 'Input2','Input3','Input4','Input5','Output1','Output2','Output3','Output4','Output5','Output6','Output7','Output8','Output9','Output10'])
        print(train.head())

        # summarize the data
        for i in range(len(train_X)):
            print(train_X[i], train_Y[i])

        # split into samples
        test_X, test_Y= split_sequence(test, n_steps_in, n_steps_out)

        test = np.concatenate((test_X, test_Y), axis=1)
        test = pd.DataFrame(test, columns = ['Input1', 'Input2','Input3','Input4','Input5','Output1','Output2','Output3','Output4','Output5','Output6','Output7','Output8','Output9','Output10'])
        print(test.head())

        # summarize the data
        for i in range(len(test_X)):
            print(test_X[i], test_Y[i])

        if problem ==1:
            train.to_csv('Lazer/train1.csv')
            test.to_csv('Lazer/test1.csv')
        if problem ==2:
            train.to_csv('Sunspot/train1.csv')
            test.to_csv('Sunspot/test1.csv')
        if problem ==3:
            train.to_csv('Mackey/train1.csv')
            test.to_csv('Mackey/test1.csv')
        if problem ==4:
            train.to_csv('Lorenz/train1.csv')
            test.to_csv('Lorenz/test1.csv')
        if problem ==5:
            train.to_csv('Rossler/train1.csv')
            test.to_csv('Rossler/test1.csv')
        if problem ==6:
            train.to_csv('Henon/train1.csv')
            test.to_csv('Henon/test1.csv')
        if problem ==7:
            train.to_csv('ACFinance/train1.csv')
            test.to_csv('ACFinance/test1.csv') 
        
if __name__ == "__main__": main()


# In[ ]:




