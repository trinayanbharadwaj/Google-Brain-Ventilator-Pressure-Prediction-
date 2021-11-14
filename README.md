# Google Brain Ventilator Pressure Prediction

The ventilator data used in this competition was produced using a modified open-source ventilator connected to an artificial bellows test lung via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue.

![ventilator architecture pic](https://user-images.githubusercontent.com/63582471/141678759-77bf93a1-e6fe-4baf-a8e8-d6fc387f2670.jpg)


# Problem Statement: 
Simulate a ventilator connected to a sedated patient's lung.

# Tools Used:
Python, Pandas, Keras, Tensorflow, Matplotlib, Seaborn, kaggle kernels

# Architecture:

1) Data provided by the competition organizer. train.csv, test.csv and submission.csv
2) Processing the train and test dataset
3) Reshaping the data into a valid format for the Bi-LSTM model
4) Build a Bi-LSTM model of 3 bidirectional layers.
5) Trained the model Early Stopping of 5 and batch size of 32
6) Saved the model
7) frehsly load the models
8) Ensembled the result of each model 
9) Created the submission file
10) Submit 

Doing heavy features engineering and data rescaling was NOT helping. Instead, the above mentioned approach gave me the best MAE. My final submission got an MAE of 0.353 in the leaderboard

![Screenshot 2021-11-14 164809](https://user-images.githubusercontent.com/63582471/141678692-2752e92a-390a-4187-bca8-bc11c8319693.jpg)

# Contributer:
Kumar Trinayan Bharadwaj
