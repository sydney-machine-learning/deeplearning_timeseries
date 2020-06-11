#!/usr/bin/env python
# coding: utf-8

# In[2]:


#source: https://github.com/sydney-machine-learning/parallel-tempering-neural-net/blob/master/multicore-pt-regression/Compare_benchmark/nn.py

import sklearn
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt

def rmse(pred, actual):
    error = np.subtract(pred, actual)
    sqerror= np.sum(np.square(error))/actual.shape[0]
    return np.sqrt(sqerror)

def main():

    for i in range(1, 8) : 
        
        problem = i
        if problem ==1:
            TrainData = pd.read_csv("../../data/Lazer/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../../data/Lazer/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "Lazer"
        if problem ==2:
            TrainData = pd.read_csv("../../data/Sunspot/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../../data/Sunspot/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "Sunspot"
        if problem ==3:
            TrainData = pd.read_csv("../../data/Mackey/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../../data/Mackey/test1.csv",index_col = 0)
            TestData = TestData.values
            name="Mackey"
        if problem ==4:
            TrainData = pd.read_csv("../../data/Lorenz/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../../data/Lorenz/test1.csv",index_col = 0)
            TestData = TestData.values  
            name= "Lorenz"
        if problem ==5:
            TrainData = pd.read_csv("../../data/Rossler/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../../data/Rossler/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "Rossler"
        if problem ==6:
            TrainData = pd.read_csv("../../data/Henon/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../../data/Henon/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "Henon"
        if problem ==7:
            TrainData = pd.read_csv("../../data/ACFinance/train1.csv",index_col = 0)
            TrainData = TrainData.values
            TestData = pd.read_csv("../../data/ACFinance/test1.csv",index_col = 0)
            TestData = TestData.values
            name= "ACFinance" 

        x_train = TrainData[:,0:5]
        y_train = TrainData[:,5:15]
        x_test = TestData[:,0:5]
        y_test = TestData[:,5:15]
        
        print(name)
        
        TrainRMSE_mean=np.zeros(3)
        TestRMSE_mean=np.zeros(3)
        TrainRMSE_Std=np.zeros(3)
        TestRMSE_Std=np.zeros(3)
        train_acc=np.zeros(30)
        test_acc=np.zeros(30)
        for run in range(30):
            mlp_adam = MLPRegressor(hidden_layer_sizes=(5, ), activation='relu', solver='adam', alpha=0.1,max_iter=5000, tol=0)
            mlp_adam.fit(x_train,y_train)
            y_predicttrain = mlp_adam.predict(x_train)
            y_predicttest = mlp_adam.predict(x_test)
            train_acc[run] = rmse( y_predicttrain, y_train) 
            test_acc[run] = rmse( y_predicttest, y_test) 
        
        arr = np.dstack((train_acc,test_acc))
        arr=arr.reshape(30,2)
        arr = pd.DataFrame(arr, index = [ 'Exp1','Exp2','Exp3','Exp4','Exp5','Exp6','Exp7','Exp8','Exp9','Exp10','Exp11','Exp12','Exp13','Exp14','Exp15','Exp16','Exp17','Exp18','Exp19','Exp20','Exp21','Exp22','Exp23','Exp24','Exp25','Exp26','Exp27','Exp28','Exp29','Exp30'], columns=['TrainRMSE','TestRMSE'])
        arr.to_csv("../../data/"+name+"/NNtimeseries_scikitearnAnalysis/mlp_adam_ExpAnalysis.csv")
        
        TrainRMSE_mean[0]=np.mean(train_acc)
        TestRMSE_mean[0]=np.mean(test_acc)
        TrainRMSE_Std[0]=np.std(train_acc)
        TestRMSE_Std[0]=np.std(test_acc)
        
        arr1 = np.vstack(([np.mean(train_acc),np.std(train_acc),np.min(train_acc),np.max(train_acc)],[np.mean(test_acc), np.std(test_acc),np.min(test_acc),np.max(test_acc)]))
        arr1 = pd.DataFrame(arr1, index=['TrainRMSE','TestRMSE'], columns = ['Mean','Standard Deviation','Min','Max'])
        print(arr1)
        arr1.to_csv("../../data/"+name+"/NNtimeseries_scikitearnAnalysis/mlp_adam_OverallAnalysis.csv")
        
        ax = plt.gca()
        arr.plot(kind='line',x=None,y='TrainRMSE',color='red', ax=ax)
        plt.xlabel('Experiments') 
        plt.savefig("../../data/"+name+"/NNtimeseries_scikitearnAnalysis/mlp_adam_TrainRMSE.png",dpi=100)
        plt.show()

        ax = plt.gca()
        arr.plot(kind='line',x=None,y='TestRMSE',color='blue', ax=ax)
        plt.xlabel('Experiments') 
        plt.savefig("../../data/"+name+"/NNtimeseries_scikitearnAnalysis/mlp_adam_TestRMSE.png",dpi=100)
        plt.show()
        
        for run in range(30):
            mlp_sgd = MLPRegressor(hidden_layer_sizes=(5, ), activation='relu', solver='sgd', alpha=0.1,max_iter=5000, tol=0)
            mlp_sgd.fit(x_train,y_train)
            y_predicttrain = mlp_sgd.predict(x_train)
            y_predicttest = mlp_sgd.predict(x_test)
            train_acc[run] = rmse( y_predicttrain,y_train) 
            test_acc[run] = rmse( y_predicttest, y_test) 
        
        arr = np.dstack((train_acc,test_acc))
        arr=arr.reshape(30,2)
        arr = pd.DataFrame(arr, index = [ 'Exp1','Exp2','Exp3','Exp4','Exp5','Exp6','Exp7','Exp8','Exp9','Exp10','Exp11','Exp12','Exp13','Exp14','Exp15','Exp16','Exp17','Exp18','Exp19','Exp20','Exp21','Exp22','Exp23','Exp24','Exp25','Exp26','Exp27','Exp28','Exp29','Exp30'], columns=['TrainRMSE','TestRMSE'])
        arr.to_csv("../../data/"+name+"/NNtimeseries_scikitearnAnalysis/mlp_sgd_ExpAnalysis.csv")
        
        TrainRMSE_mean[1]=np.mean(train_acc)
        TestRMSE_mean[1]=np.mean(test_acc)
        TrainRMSE_Std[1]=np.std(train_acc)
        TestRMSE_Std[1]=np.std(test_acc)
        
        arr1 = np.vstack(([np.mean(train_acc),np.std(train_acc),np.min(train_acc),np.max(train_acc)],[np.mean(test_acc), np.std(test_acc),np.min(test_acc),np.max(test_acc)]))
        arr1 = pd.DataFrame(arr1, index=['TrainRMSE','TestRMSE'], columns = ['Mean','Standard Deviation','Min','Max'])
        print(arr1)
        arr1.to_csv("../../data/"+name+"/NNtimeseries_scikitearnAnalysis/mlp_sgd_OverallAnalysis.csv")
        
        ax = plt.gca()
        arr.plot(kind='line',x=None,y='TrainRMSE',color='red', ax=ax)
        plt.xlabel('Experiments') 
        plt.savefig("../../data/"+name+"/NNtimeseries_scikitearnAnalysis/mlp_sgd_TrainRMSE.png",dpi=100)
        plt.show()

        ax = plt.gca()
        arr.plot(kind='line',x=None,y='TestRMSE',color='blue', ax=ax)
        plt.xlabel('Experiments') 
        plt.savefig("../../data/"+name+"/NNtimeseries_scikitearnAnalysis/mlp_sgd_TestRMSE.png",dpi=100)
        plt.show()
        
        for run in range(30):
            rf = RandomForestRegressor()
            rf.fit(x_train,y_train)
            y_predicttrain = rf.predict(x_train)
            y_predicttest = rf.predict(x_test)
            train_acc[run] = rmse( y_predicttrain,y_train) 
            test_acc[run] = rmse( y_predicttest, y_test)
        
        arr = np.dstack((train_acc,test_acc))
        arr=arr.reshape(30,2)
        arr = pd.DataFrame(arr, index = [ 'Exp1','Exp2','Exp3','Exp4','Exp5','Exp6','Exp7','Exp8','Exp9','Exp10','Exp11','Exp12','Exp13','Exp14','Exp15','Exp16','Exp17','Exp18','Exp19','Exp20','Exp21','Exp22','Exp23','Exp24','Exp25','Exp26','Exp27','Exp28','Exp29','Exp30'], columns=['TrainRMSE','TestRMSE'])
        arr.to_csv("../../data/"+name+"/NNtimeseries_scikitearnAnalysis/RF_ExpAnalysis.csv")
        
        TrainRMSE_mean[2]=np.mean(train_acc)
        TestRMSE_mean[2]=np.mean(test_acc)
        TrainRMSE_Std[2]=np.std(train_acc)
        TestRMSE_Std[2]=np.std(test_acc)
        
        arr1 = np.vstack(([np.mean(train_acc),np.std(train_acc),np.min(train_acc),np.max(train_acc)],[np.mean(test_acc), np.std(test_acc),np.min(test_acc),np.max(test_acc)]))
        arr1 = pd.DataFrame(arr1, index=['TrainRMSE','TestRMSE'], columns = ['Mean','Standard Deviation','Min','Max'])
        print(arr1)
        arr1.to_csv("../../data/"+name+"/NNtimeseries_scikitearnAnalysis/RF_OverallAnalysis.csv")
        
        ax = plt.gca()
        arr.plot(kind='line',x=None,y='TrainRMSE',color='red', ax=ax)
        plt.xlabel('Experiments') 
        plt.savefig("../../data/"+name+"/NNtimeseries_scikitearnAnalysis/RF_TrainRMSE.png",dpi=100)
        plt.show()

        ax = plt.gca()
        arr.plot(kind='line',x=None,y='TestRMSE',color='blue', ax=ax)
        plt.xlabel('Experiments') 
        plt.savefig("../../data/"+name+"/NNtimeseries_scikitearnAnalysis/RF_TestRMSE.png",dpi=100)
        plt.show()
        
        labels = ['TrainRMSE_Mean','TestRMSE_Mean']
        MLP_Adam=[TrainRMSE_mean[0],TestRMSE_mean[0]]
        MLP_Sgd=[TrainRMSE_mean[1],TestRMSE_mean[1]]
        RF=[TrainRMSE_mean[2],TestRMSE_mean[2]]
        width = 0.25  # the width of the bars
        r1 = np.arange(len(labels))
        r2 = [x + width for x in r1]
        r3 = [x + width for x in r2]
        
        fig, ax = plt.subplots()
        rects1 = ax.bar(r1, MLP_Adam, width, label='MLP_Adam')
        rects2 = ax.bar(r2, MLP_Sgd, width, label='MLP_Sgd')
        rects3 = ax.bar(r3, RF, width, label='RandomForest')
        
        ax.set_title('MLP_Adam vs MLP_Sgd vs RandonForest')
        plt.xticks([r + width for r in range(len(MLP_Adam))], ['TrainRMSE_Mean', 'TestRMSE_Mean'])
        ax.legend()
        fig.tight_layout()
        plt.savefig("../../data/"+name+"/NNtimeseries_scikitearnAnalysis/TrainRMSE_Comparison.png",dpi=100)
        plt.show()
        
        
        labels = ['TrainRMSE_Std','TestRMSE_Std']
        MLP_Adam=[TrainRMSE_Std[0],TestRMSE_Std[0]]
        MLP_Sgd=[TrainRMSE_Std[1],TestRMSE_Std[1]]
        RF=[TrainRMSE_Std[2],TestRMSE_Std[2]]
        width = 0.25  # the width of the bars
        r1 = np.arange(len(labels))
        r2 = [x + width for x in r1]
        r3 = [x + width for x in r2]
        
        fig, ax = plt.subplots()
        rects1 = ax.bar(r1, MLP_Adam, width, label='MLP_Adam')
        rects2 = ax.bar(r2, MLP_Sgd, width, label='MLP_Sgd')
        rects3 = ax.bar(r3, RF, width, label='RandomForest')
        
        ax.set_title('MLP_Adam vs MLP_Sgd vs RandonForest')
        plt.xticks([r + width for r in range(len(MLP_Adam))], ['TrainRMSE_Std', 'TestRMSE_Std'])
        ax.legend()
        fig.tight_layout()
        plt.savefig("../../data/"+name+"/NNtimeseries_scikitearnAnalysis/TestRMSE_Comparison.png",dpi=100)
        plt.show()

if __name__ == "__main__": main()


# In[ ]:




