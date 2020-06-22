#!/usr/bin/env python
# coding: utf-8

# In[1]:


from math import sqrt
import numpy as np
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pandas import read_csv
from datetime import datetime
from keras.layers import Bidirectional
import datetime
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from numpy import array
import time


# In[2]:


def rmse(pred, actual):
    error = np.subtract(pred, actual)
    sqerror= np.sum(np.square(error))/actual.shape[0]
    return np.sqrt(sqerror)


# In[3]:


def MODEL_FNN_adam(x_train,x_test,y_train,y_test,Num_Exp,n_steps_in,n_steps_out,Iterations):
    train_acc=np.zeros(Num_Exp)
    test_acc=np.zeros(Num_Exp)
    Step_RMSE=np.zeros([Num_Exp,n_steps_out])
    mlp_adam = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.1,max_iter=Iterations, tol=0)
    
    start_time=time.time()
    for run in range(Num_Exp):
        print("Experiment",run+1,"in progress")
        mlp_adam.fit(x_train, y_train)
        y_predicttrain = mlp_adam.predict(x_train)
        y_predicttest = mlp_adam.predict(x_test)
        train_acc[run] = rmse( y_predicttrain, y_train) 
        test_acc[run] = rmse( y_predicttest, y_test)
        for j in range(n_steps_out):
            Step_RMSE[run][j]=rmse(y_predicttest[:,j], y_test[:,j])
            
    print("Total time for",Num_Exp,"experiments",time.time()-start_time)
    return train_acc,test_acc,Step_RMSE

def MODEL_FNN_sgd(x_train,x_test,y_train,y_test,Num_Exp,n_steps_in,n_steps_out,Iterations):
    train_acc=np.zeros(Num_Exp)
    test_acc=np.zeros(Num_Exp)
    Step_RMSE=np.zeros([Num_Exp,n_steps_out])
    mlp_sgd = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='sgd', alpha=0.1,max_iter=Iterations, tol=0)
    
    start_time=time.time()
    for run in range(Num_Exp):
        print("Experiment",run+1,"in progress")
        mlp_sgd.fit(x_train,y_train)
        y_predicttrain = mlp_sgd.predict(x_train)
        y_predicttest = mlp_sgd.predict(x_test)
        train_acc[run] = rmse( y_predicttrain,y_train) 
        test_acc[run] = rmse( y_predicttest, y_test) 
        for j in range(n_steps_out):
            Step_RMSE[run][j]=rmse(y_predicttest[:,j], y_test[:,j])
            
    print("Total time for",Num_Exp,"experiments",time.time()-start_time)
    return train_acc,test_acc,Step_RMSE


def MODEL_LSTM(x_train,x_test,y_train,y_test,Num_Exp,n_steps_in,n_steps_out,Iterations):
    n_features = 1
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
    print(x_train.shape)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
    print(x_test.shape)
    
    train_acc=np.zeros(Num_Exp)
    test_acc=np.zeros(Num_Exp)
    Step_RMSE=np.zeros([Num_Exp,n_steps_out])
    
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in,n_features)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    start_time=time.time()
    for run in range(Num_Exp):
        print("Experiment",run+1,"in progress")
        # fit model
        model.fit(x_train, y_train, epochs=Iterations,batch_size=64, verbose=0, shuffle=False)
        y_predicttrain = model.predict(x_train)
        y_predicttest = model.predict(x_test)
        train_acc[run] = rmse( y_predicttrain,y_train) 
        test_acc[run] = rmse( y_predicttest, y_test) 
        for j in range(n_steps_out):
            Step_RMSE[run][j]=rmse(y_predicttest[:,j], y_test[:,j])
            
    print("Total time for",Num_Exp,"experiments",time.time()-start_time)
    return train_acc,test_acc,Step_RMSE


def MODEL_Bi_LSTM(x_train,x_test,y_train,y_test,Num_Exp,n_steps_in,n_steps_out,Iterations):
    n_features = 1
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
    print(x_train.shape)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
    print(x_test.shape)
    
    train_acc=np.zeros(Num_Exp)
    test_acc=np.zeros(Num_Exp)
    Step_RMSE=np.zeros([Num_Exp,n_steps_out])
    
    model = Sequential()
    model.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(n_steps_in,n_features)))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    start_time=time.time()
    for run in range(Num_Exp):
        print("Experiment",run+1,"in progress")
        # fit model
        model.fit(x_train, y_train, epochs=Iterations,batch_size=64, verbose=0, shuffle=False)
        y_predicttrain = model.predict(x_train)
        y_predicttest = model.predict(x_test)
        train_acc[run] = rmse( y_predicttrain,y_train) 
        test_acc[run] = rmse( y_predicttest, y_test) 
        for j in range(n_steps_out):
            Step_RMSE[run][j]=rmse(y_predicttest[:,j], y_test[:,j])
            
    print("Total time for",Num_Exp,"experiments",time.time()-start_time)
    return train_acc,test_acc,Step_RMSE


def MODEL_EN_DC(x_train,x_test,y_train,y_test,Num_Exp,n_steps_in,n_steps_out,Iterations):
    n_features = 1
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
    print(x_train.shape)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
    print(x_test.shape)
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
    print(y_train.shape)
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))
    print(y_test.shape)
    
    train_acc=np.zeros(Num_Exp)
    test_acc=np.zeros(Num_Exp)
    Step_RMSE=np.zeros([Num_Exp,n_steps_out])
    
    model = Sequential()
    model.add(LSTM(100, activation='relu',input_shape=(n_steps_in,n_features)))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1,activation='relu')))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    start_time=time.time()
    for run in range(Num_Exp):
        print("Experiment",run+1,"in progress")
        # fit model
        model.fit(x_train, y_train, epochs=Iterations,batch_size=64, verbose=0, shuffle=False)
        y_predicttrain = model.predict(x_train)
        y_predicttest = model.predict(x_test)
        train_acc[run] = rmse( y_predicttrain,y_train) 
        test_acc[run] = rmse( y_predicttest, y_test) 
        for j in range(n_steps_out):
            Step_RMSE[run][j]=rmse(y_predicttest[:,j,0], y_test[:,j,0])
            
    print("Total time for",Num_Exp,"experiments",time.time()-start_time)
    return train_acc,test_acc,Step_RMSE


# In[4]:


def Plot_Mean(name,TrainRMSE_mean,TestRMSE_mean):
    labels = ['TrainRMSE','TestRMSE']
    MLP_Adam=[TrainRMSE_mean[0],TestRMSE_mean[0]]
    MLP_Sgd=[TrainRMSE_mean[1],TestRMSE_mean[1]]
    LSTM=[TrainRMSE_mean[2],TestRMSE_mean[2]]
    Bi_LSTM=[TrainRMSE_mean[3],TestRMSE_mean[3]]
    EN_DC=[TrainRMSE_mean[4],TestRMSE_mean[4]]
    width = 0.1  # the width of the bars
    Plot(name,labels,width,MLP_Adam,MLP_Sgd,LSTM,Bi_LSTM,EN_DC,"Mean","Train&Test_RMSE_Mean_Comparison")


def Plot_Std(name,TrainRMSE_Std,TestRMSE_Std):
    labels = ['TrainRMSE','TestRMSE']
    MLP_Adam=[TrainRMSE_Std[0],TestRMSE_Std[0]]
    MLP_Sgd=[TrainRMSE_Std[1],TestRMSE_Std[1]]
    LSTM=[TrainRMSE_Std[2],TestRMSE_Std[2]]
    Bi_LSTM=[TrainRMSE_Std[3],TestRMSE_Std[3]]
    EN_DC=[TrainRMSE_Std[4],TestRMSE_Std[4]]
    width = 0.1  # the width of the bars
    Plot(name,labels,width,MLP_Adam,MLP_Sgd,LSTM,Bi_LSTM,EN_DC,"Standard Deviation","Train&Test_RMSE_Std_Comparison")

def Plot_Step_RMSE_Mean(name,Step_RMSE_mean):
    MLP_Adam=Step_RMSE_mean[0,:]
    MLP_Sgd=Step_RMSE_mean[1,:]
    LSTM=Step_RMSE_mean[2,:]
    Bi_LSTM=Step_RMSE_mean[3,:]
    EN_DC=Step_RMSE_mean[4,:]
    labels = []
    for j in range(Step_RMSE_mean.shape[1]):
        labels=np.concatenate((labels,[str(j+1)]))
    width = 0.1  # the width of the bars
    Plot(name,labels,width,MLP_Adam,MLP_Sgd,LSTM,Bi_LSTM,EN_DC,"RMSE_Mean","Step_RMSE_Comparison")
    

def Plot(name,labels,width,MLP_Adam,MLP_Sgd,LSTM,Bi_LSTM,EN_DC,typ,Gname):
    r1 = np.arange(len(labels))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r4 = [x + width for x in r3]
    r5 = [x + width for x in r4]

    fig, ax = plt.subplots()
    rects1 = ax.bar(r1, MLP_Adam, width, label='MLP_Adam')
    rects2 = ax.bar(r2, MLP_Sgd, width, label='MLP_Sgd')
    rects3 = ax.bar(r3, LSTM, width, label='LSTM')
    rects4 = ax.bar(r4, Bi_LSTM, width, label='Bi_LSTM')
    rects5 = ax.bar(r5, EN_DC, width, label='EN_DC')

    plt.ylabel(typ)
    ax.set_title('MLP_Adam vs MLP_Sgd vs Stacked LSTM vs Bi-LSTM vs EN-DC LSTM')
    plt.xticks([r + width for r in range(len(MLP_Adam))], labels)
    ax.legend()
    fig.tight_layout()
    plt.savefig("Results/"+name+"/"+Gname+".png",dpi=100)
    plt.show()
    


# In[ ]:


def main():
    
    n_steps_in, n_steps_out = 5,10
    
    Overall_Analysis=np.zeros([35,6+n_steps_out*3])
    for i in range(1,8):
        problem=i
        if problem ==1:
            TrainData = pd.read_csv("../data/Lazer/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../data/Lazer/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "Lazer"
        if problem ==2:
            TrainData = pd.read_csv("../data/Sunspot/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../data/Sunspot/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "Sunspot"
        if problem ==3:
            TrainData = pd.read_csv("../data/Mackey/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../data/Mackey/test1.csv",index_col = 0)
            TestData = TestData.values
            name="Mackey"
        if problem ==4:
            TrainData = pd.read_csv("../data/Lorenz/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../data/Lorenz/test1.csv",index_col = 0)
            TestData = TestData.values  
            name= "Lorenz"
        if problem ==5:
            TrainData = pd.read_csv("../data/Rossler/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../data/Rossler/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "Rossler"
        if problem ==6:
            TrainData = pd.read_csv("../data/Henon/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../data/Henon/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "Henon"
        if problem ==7:
            TrainData = pd.read_csv("../data/ACFinance/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../data/ACFinance/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "ACFinance" 

        x_train = TrainData[:,0:n_steps_in]
        y_train = TrainData[:,n_steps_in : n_steps_in+n_steps_out ]
        x_test = TestData[:,0:n_steps_in]
        y_test = TestData[:,n_steps_in : n_steps_in+n_steps_out]

        print(name)
        Num_Exp=5    #No. of experiments
        Iterations=500
        TrainRMSE_mean=np.zeros(5)
        TestRMSE_mean=np.zeros(5)
        TrainRMSE_Std=np.zeros(5)
        TestRMSE_Std=np.zeros(5)
        Step_RMSE_mean=np.zeros([5,n_steps_out])
        train_acc=np.zeros(Num_Exp)
        test_acc=np.zeros(Num_Exp)
        Step_RMSE=np.zeros([Num_Exp,n_steps_out])


        for k in range(1,6):

            method=k
            if method ==1:
                train_acc,test_acc,Step_RMSE=MODEL_FNN_adam(x_train,x_test,y_train,y_test,Num_Exp,n_steps_in,n_steps_out,Iterations)
                Mname="MODEL_FNN_adam"
            if method ==2:
                train_acc,test_acc,Step_RMSE=MODEL_FNN_sgd(x_train,x_test,y_train,y_test,Num_Exp,n_steps_in,n_steps_out,Iterations)
                Mname="MODEL_FNN_sgd"
            if method ==3:
                train_acc,test_acc,Step_RMSE=MODEL_LSTM(x_train,x_test,y_train,y_test,Num_Exp,n_steps_in,n_steps_out,Iterations)
                Mname="MODEL_LSTM"
            if method ==4:
                train_acc,test_acc,Step_RMSE=MODEL_Bi_LSTM(x_train,x_test,y_train,y_test,Num_Exp,n_steps_in,n_steps_out,Iterations)
                Mname="MODEL_Bi_LSTM"
            if method ==5:
                train_acc,test_acc,Step_RMSE=MODEL_EN_DC(x_train,x_test,y_train,y_test,Num_Exp,n_steps_in,n_steps_out,Iterations)
                Mname="MODEL_EN_DC"

            print(Mname)

            Overall_Analysis[(i-1)*5+(k-1)][0]=np.mean(train_acc)
            Overall_Analysis[(i-1)*5+(k-1)][1]=np.std(train_acc)
            Overall_Analysis[(i-1)*5+(k-1)][2]=np.min(train_acc)
            Overall_Analysis[(i-1)*5+(k-1)][3]=np.mean(test_acc)
            Overall_Analysis[(i-1)*5+(k-1)][4]=np.std(test_acc)
            Overall_Analysis[(i-1)*5+(k-1)][5]=np.min(test_acc)

            arr = np.dstack((train_acc,test_acc))
            arr=arr.reshape(Num_Exp,2)
            arr=np.concatenate((arr,Step_RMSE), axis=1)
            arr=arr.reshape(Num_Exp,2+n_steps_out)
            ExpIndex=np.array([])
            for j in range(Num_Exp):
                ExpIndex=np.concatenate((ExpIndex,["Exp"+str(j+1)]))
            
            TrainRMSE_mean[k-1]=np.mean(train_acc)
            TestRMSE_mean[k-1]=np.mean(test_acc)
            TrainRMSE_Std[k-1]=np.std(train_acc)
            TestRMSE_Std[k-1]=np.std(test_acc)
            
            ExpIndex1=['TrainRMSE','TestRMSE']
            for j in range(n_steps_out):
                Step_RMSE_mean[k-1][j]=np.mean(Step_RMSE[:,j])
                ExpIndex1=np.concatenate((ExpIndex1,["Step"+str(j+1)]))
            arr = pd.DataFrame(arr, index = ExpIndex , columns = ExpIndex1)
            arr.to_csv("Results/"+name+"/"+Mname+"/ExpAnalysis.csv")
            print(arr)
            
            arr1 = np.vstack(([np.mean(train_acc),np.std(train_acc),np.min(train_acc),np.max(train_acc)],[np.mean(test_acc), np.std(test_acc),np.min(test_acc),np.max(test_acc)]))
            for j in range(n_steps_out):
                Step_mean = np.mean(Step_RMSE[:,j])
                Step_std = np.std(Step_RMSE[:,j])
                Step_min = np.min(Step_RMSE[:,j])
                arr1=np.vstack((arr1,[Step_mean,Step_std,Step_min,np.max(Step_RMSE[:,j])]))
                Overall_Analysis[(i-1)*5+(k-1)][3*j+6]= Step_mean
                Overall_Analysis[(i-1)*5+(k-1)][3*j+7]= Step_std
                Overall_Analysis[(i-1)*5+(k-1)][3*j+8]= Step_min
            arr1 = pd.DataFrame(arr1, index=ExpIndex1, columns = ['Mean','Standard Deviation','Min','Max'])
            print(arr1)
            arr1.to_csv("Results/"+name+"/"+Mname+"/OverallAnalysis.csv")
        
        
        
        #Plot mean of train_RMSE and test_RMSE
        Plot_Mean(name,TrainRMSE_mean,TestRMSE_mean)
        #Plot Std of train_RMSE and test_RMSE
        Plot_Std(name,TrainRMSE_Std,TestRMSE_Std)
        #Plot Step wise RMSE mean for different methods
        Plot_Step_RMSE_Mean(name,Step_RMSE_mean)

        
    Index1=[]
    for j in range(7):
        Index1=np.concatenate((Index1, ['MLP_Adam','MLP_Sgd','Stacked_LSTM','Bi-LSTM','EN-DC LSTM']))
    Index2=["Lazer","Sunspot","Mackey","Lorenz","Rossler","Henon","ACFinance"]
    Index2=np.repeat(Index2,5)
    Index=np.dstack((Index2,Index1))
    Index=Index.reshape(35,2)
    Column=['Dataset','Method','TrainRMSE_Mean','TrainRMSE_Std','TrainRMSE_Min','TestRMSE_Mean','TestRMSE_Std','TestRMSE_Min']
    for j in range(1,11):
        Column=np.concatenate((Column, ['Step'+str(j)+'RMSE_Mean','Step'+str(j)+'RMSE_Std','Step'+str(j)+'RMSE_Min']))
  
    Overall_Analysis=np.concatenate((Index,Overall_Analysis), axis=1)
    Overall_Analysis = pd.DataFrame(Overall_Analysis, columns = Column)
    print(Overall_Analysis)
    Overall_Analysis.to_csv("Results/OverallAnalysis.csv")
    
    
    
if __name__ == "__main__": main()


# In[ ]:




