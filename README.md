# Deeplearning_timeseries
Evaluation of shallow and deep learning models for multi-step-ahead time series prediction

We present an evaluation study that compares the performance of deep learning models for multi-step ahead time series prediction. The deep learning methods comprise simple recurrent neural networks, long-short-term memory (LSTM) networks, bidirectional LSTM networks, encoder-decoder LSTM networks,and convolutional neural networks. We provide a further comparison with simple neural networks that use stochastic gradient descent and adaptive moment estimation (Adam) for training. We focus on univariate time series for multi-step-ahead prediction from benchmark time-series datasets and provide a further comparison of the results with related methods from the literature.

## Code
We have a unified code for all datasets with proper comments.
* The python notebook for implementation can be found here: [Code](https://github.com/sydney-machine-learning/deeplearning_timeseries/blob/master/FNN/Code.ipynb)
  
## Data
We use a combination of benchmark problems that include simulated and real-world  time  series. The simulated time series are Mackey-Glass, Lorenz, Henon, and Rossler. The real-world time series are Sunspot, Lazer and ACI-financial time series. 
The dataset used in experiments can be found here: [Data](https://github.com/sydney-machine-learning/deeplearning_timeseries/tree/master/data)
Also we had to pre-process the datasets for our experiments.
* The python notebook for preprocessing can be found here: [Preprocessing](https://github.com/sydney-machine-learning/deeplearning_timeseries/blob/master/data/Data%20processing.ipynb)

## Experiments
Overall results (30 runs) for different datasets using different models and training strategies can be found here: [Results](https://github.com/sydney-machine-learning/deeplearning_timeseries/blob/master/FNN/Results/OverallAnalysis.csv)
