### README.md

 - Hello!
 - This is an experimental project to compare COVID-19 daily/weekly confirmed patient prediction accuracy using various deep learning models.
 - Things will be updated frequently.






# 1. Data


#### My data are divided into two main categories. The number of confirmed COVID_19 patients and crawled data from news articles every day in Korea.





### 1.1. The number of confirmed COVID_19 patients.

1. This Data consists of the number of confirmed COVID_19 patients everyday in South Korea. 

2. All data were obtained from the [Korea Data Exchange](https://kdx.kr/main).

3. From February 21, 2020, when the epidemic began to spread in earnest, data was collected and obtained until August 3, 2021.

4. I've transformed this data into the number of new confirmed cases per day, as there were only cumulative values of Korean COVID_19 cases in it.

> ![image](https://user-images.githubusercontent.com/50206883/128275492-f229e8bc-08c7-445b-a83b-1afed48d9114.png)





### 1.2. COVID_19 articles collected daily by [Naver](https://naver.com), a large Korean portal site.

1. There are several factors affecting the number of new confirmed cases of COVID_19.

2. One is data on the number of daily confirmed cases so far, and the other is whether there is a cluster of COVID_19 cases and the transmission pattern.

3. Since I could not arbitrarily judge the latter one, I have decided to collect daily COVID_19 articles and analyze them to judge it.


> ![image](https://user-images.githubusercontent.com/50206883/128275695-7a13624b-fb8b-4103-9a20-dce5ba8dea7a.png)
> ![image](https://user-images.githubusercontent.com/50206883/128272941-55dca9e0-29b0-4a79-bc3f-55c7daaf7676.png)
> ![image](https://user-images.githubusercontent.com/50206883/128272963-3bbdadfd-7315-4fb1-a819-77fe6dea4d15.png)





### 1.3. Processed data



1. I first made a list of positive and negative keywords.

> Positive keywords
>
> ![image](https://user-images.githubusercontent.com/50206883/129861520-760dfa79-bfac-4901-b643-229d7d2f4f51.png)



> Negative keywords
>
> ![image](https://user-images.githubusercontent.com/50206883/129861175-3511903c-a9b6-47b4-b6f6-cc4c2ba589c3.png)




2. In each of the 100 articles crawled every day, I score -1 if there is a negative word, +1 if there is a positive word, and 0 if both words come out or neither comes out. That is, the daily score can be ranged from -100 to +100.

> An example
> 
> ![image](https://user-images.githubusercontent.com/50206883/129863831-98fda5a1-b548-456b-9d2f-3048a869f350.png)
> 
> Score
> 
> ![image](https://user-images.githubusercontent.com/50206883/129864279-1ba99325-3996-4f0f-9b41-3229abc782a6.png)




3. I then bounded the number of confirmed coronavirus cases per day, crawled keyword scores, and symbol for whether the day is not a legal holiday(0 if it is, 1 if not) and made it into a set.


> An example
>
> ![image](https://user-images.githubusercontent.com/50206883/130180577-98bd5969-942e-4f10-9e7e-b26ee7d76e58.png)

4. The reason why the last data was not one-hot-encoded is that the daily number of coronavirus 19 patients is usually larger than the legal holiday on non-legal days, so it is expressed as a number greater than zero to reflect the weight of input in artificial intelligence.


5. I used seven consecutive bundles of these data as an input.

> An example of an input
>
> ![image](https://user-images.githubusercontent.com/50206883/130181414-f2524567-992d-40cb-aa84-0d78b4fc378e.png)



### 1.4. Target


1. I then set the average daily number of confirmed coronavirus cases for the next week as the target of the input data.


> An example of a target
>
> ![image](https://user-images.githubusercontent.com/50206883/130182552-1fa00931-9903-4bd2-b092-f80a8228f814.png)






# 2. Algorithm


### 2.0. Prediction and Forecasting

1. I mainly had two tasks. Prediction and Forecasting.

2. For Forecasting I can use variety of AI models, but since I cannot use sequential AI model for Prediction tasks, the available models for it is very limited.

### 2.1. Acessing the Data


1. All of my data are saved in my [Google Drive](https://www.google.com/intl/ko_KR/drive/)

2. I accessed to these data using Pandas' read_cvs method.


> The code
>
> ![image](https://user-images.githubusercontent.com/50206883/130183429-1a1a3e5f-7c29-412e-9358-a174628bb5ae.png)


### 2.2. Splitting the Data

1. If the task were prediction I shuffled the data when splitting it into training data and test data.

2. If the task were forecasting I did not shuffled the data.

3. The test size is 0.15 which is built from daily number of confirmed coronavirus cases from 2021/5/5 to 2021/08/03 which contains peroid of 4th Pandemic of COVID_19 in south korea.

> An example
>
> ![image](https://user-images.githubusercontent.com/50206883/130186053-f9233195-cbb4-450e-b830-dc7019a87dee.png)

### 2.3. Pre-processing data

#### 2.3.1. Prediction task


#### 2.3.2. Forecasting task

1. I used MaxAbsScaler as a scaler, which can reflect negative numbers very well, because there is a lot of negative data that is processed by crawl.

2. I used Random Search CV to find the parameters of the deep learning model, and I did not preprocess train data because I took scalar based on the training data that excludes the Validation data that changes every time in the pipeline of Random Search CV.

3. When all the Random Search CV processes were terminated and the optimal hyperparameter found by the Random Search CV had to be fit into the new model, the data used at this time had to be scalarized in advance because the pipeline did not exist.

4. The data used in this case was allocated separately as train_input_forlast.

5. Since sequential models receive only three-dimensional data as inputs, and ANN and DNN receive two-dimensional inputs, inputs for ANN and DNN are separated from train_input_forlast and assigned separately as train_input_forlast_ANN.

6. Targets used in sequential models do not require the process of converting arrays into three dimensions, so sequential models and ANN/DNN models can use the same format of arrays, but I have nonetheless assigned '_ANN' at the end for code consistency.

7. When using sequential models, I also pre-processed the train target to solve the problem that the target's scale does not match the scale of input's daily corona 19 confirmed cases data.

8. The test data was also preprocessed using a scalar from train through the same process as the train data.

> An example
>
> ![image](https://user-images.githubusercontent.com/50206883/130187386-09bf91a3-c8e9-40e6-bfe1-2b85c5bec0ed.png)


### 2.4. Build function for variety of AI models

#### 2.4.1. Prediction task


#### 2.4.2. Forecasting task


1. I used 14 A.I.s for the Forecasting task and defined functions for building each model.

2. I designed the function to be able to pass parameters to apply hyperparameters that I pass when I use Random Search CV.

3. 14 A.I models I used are "ANN with 0 hidden layers", "ANN with 1 hidden layers", "DNN with 2 hidden layers", "DNN with 3 hidden layers", "Simple RNN", "LSTM", "GRU", "LSTM with one DENSE layers", "GRU with one DENSE layers", "1D CNN with LSTM", "1D CNN with GRU", "LSTM with LSTM", "GRU with GRU", "LSTM with GRU"


4. They can be divided into six categories. ANNs, DNNs, Sequential Models, Sequential Model with Dense layers, 1D CNN with Sequential Models, Sequential with Sequential Models.

> ANNs
>
> ANN with 0 hidden layers
>> 
>> I had the Random Search CV set each value of the first neuron count, Drop_val, optimizer val, l1 regulation and l2 regulation.
>> 
>> Then I added BatchNormalization, to normalize the batch.
>>
>> I compiled the model's loss as MSE(Mean Square Error)
>>
>> Every optimizer val is either rmsprop or adam
>> 
>> ![image](https://user-images.githubusercontent.com/50206883/130196385-a26959f1-e27a-414e-821b-bf82c44b2a9e.png)
>>
> ANN with 1 hidden layers  
>>
>> Basically, the mechanism for subsequent ANNs/DNNs is the same, except that the Dense layers are added one by one as the model becomes more complex, and the Hyperparameters of these layers are tuned independently by RandomSearchCV.
>> 
>> ![image](https://user-images.githubusercontent.com/50206883/130221678-837470df-ffa9-40b6-ad8d-900fccc387bd.png)
>
> DNNs
>
> DNN with 2 hidden layers  
>>
>> ![image](https://user-images.githubusercontent.com/50206883/130311727-16aca741-1a06-4232-a165-3590e06b6dc0.png)
>
> DNN with 3 hidden layers  
>>
>> ![image](https://user-images.githubusercontent.com/50206883/130311757-289230a6-2c7c-4648-b553-bccb87f5bbde.png)
>>
> Sequential Models
> 
> Simple RNN
>>
>>I created a Reshape layer that transforms Input data into a two-dimensional array because Sequential Models have to be two-dimensional.
>>
>> Unlike other Sequential Models, Simple RNN uses activation Hyper parameter rather than recurrent_activation Hyper parameter.
>>
>> The number of neurons, the l1 and l2 values of the recurrent_regularizer, activator, optimizer, current dropout, and dropout were all determined by the Random Search CV.
>>
>> ![image](https://user-images.githubusercontent.com/50206883/130312837-113f4313-dd1c-4052-af1f-f14fb008bc08.png)
>>
> LSTM&GRU
>>
>> LSTM and GRU have the same basic mechanism as Simple RNN, except that they have recurrent_activation and that it is used instead of activation.
>> 
>> ![image](https://user-images.githubusercontent.com/50206883/130314587-f1fe1ee1-5852-44f2-90f0-1069423ea7de.png)
>>
> Sequential Model with Dense layers
>
> LSTM&GRU with one DENSE layers
>>
>> This has a Dense layer attached to the LSTM/GRU model, which has the same structure as that used in ANN/DNN.
>>
>> In the same way as the models above, Hyper parameters on all layers are found independently by Random Search CV.
>>
>>![image](https://user-images.githubusercontent.com/50206883/130315442-d1ba2978-9d62-40a5-b4d2-1193a47f518b.png)
>>
> 1D CNN with Sequential Models
>
> 1D CNN with LSTM&GRU
>>
>> I used 1D_CNN, which is good for analyzing patterns in Time series, in combination with Sequential Model.
>>
>> I had the Random Search CV set CNN's neuron count, Window size, activation, and MaxPooling values.
>>
>>I set the padding to same so that the size of the data transferred to the subsequent Sequential layer is the same as the initial size.
>>
>>![image](https://user-images.githubusercontent.com/50206883/130317050-43906a85-b779-4a90-bc8b-ae5562952724.png)
>>
> Sequential with Sequential Models
>
>
>> LSTM with LSTM, GRU with GRU, LSTM with GRU
>>
>>This basically has two Sequential layers, but the difference is that the return_sequences of the first layer were set to True to deliver the data to the second layer in the same format.
>>
>>![image](https://user-images.githubusercontent.com/50206883/130317336-03d09154-c9ec-4c04-80b4-80e456846505.png)
>>![image](https://user-images.githubusercontent.com/50206883/130317357-2fe0f6f8-657c-40d2-a888-5e1fa7f83a23.png)

### 2.5. Functions for Random Search CV

#### 2.5.1. Prediction task


#### 2.5.2. Forecasting task


1. Call Function

> I first ran the Start_Random_Search function that runs the Random Search CV to store the fitted Random Search CV in the grid variable.
> 
> Then I printed out the best Hyper Parameter found by Random Search CV and checked it.
>
> Then I called the Make_Parameter_Epochs function, which separates the Epoch from the rest of the optimal HyperParameters found by Random Search CV, and saved them separately.
>
> Then I created and trained a new model based on this Hyper Parameter, drew a plot, and returned validation RMSE and test RMSE, MAE.
>
> ![image](https://user-images.githubusercontent.com/50206883/130318412-4fd68a7c-a03d-455b-89ac-4343caf905ad.png)

2. Start_Random_Search function

> First, Keras Regressor was created with the build function that was delivered.
>
> I then made Pipeline to take MaxAbsScaler with train data except for Validation Data.
>
> I then ran RandomizedSearchCV with pipeline and a given parameter distribution.
>
> To create enough models, n_iter is set to 50.
>
> Sequential models use TimeSeries Split as CV because the validation data must be newer than train data. Therefore, I set the CV to be delivered to parameters because I have to choose differently depending on the type of model.
>
> This function then returns the Random Search CV it created.
>
>![image](https://user-images.githubusercontent.com/50206883/130318819-b0676717-ef0d-47cf-b863-0efbed84504b.png)

3. Make_Paramter_Epochs function

>
> Among the optimal hyperparameter dictionary values found by Random Search CV, non-epoch data were stored in Parameter_list and epoch in epochs_val.
>
> Because HyperParameter's name is saved based on Model_Name_Short stored in Pipeline (for example, if Model_Name_Short is 'ANN' and epoch is saved, it is saved as ANN__epoch). I have defined the function that creates the epoch's name regarding Model_Name_Short.
>
> ![image](https://user-images.githubusercontent.com/50206883/130318936-0b811df5-fd70-4265-b252-77feee36b320.png)

4. Make_Epoch_Name function

> This creates the epoch's name.
>
> ![image](https://user-images.githubusercontent.com/50206883/130319242-e8db104a-7b45-43a5-bf75-0439b22f09fd.png)


5. Draw_Plot function

> This is the function that draws plot.
>
> ![image](https://user-images.githubusercontent.com/50206883/130319269-9ccc4bcb-1218-46ba-891b-940ff79789ef.png)


### 2.6. Parameter distribution for every models.

#### 2.6.1. Prediction task


#### 2.6.2. Forecasting task

1. Non-Sequential Models(ANNs and DNNs)

> I set the range of epoch values from 250 to 400, based on a series of experiments with the same remaining conditions for Non - Sequential Models, which resulted in better performance with the epoch values from 250 to 400.


> I made sure that the activator was selected between tanh and sigmoid, which can reflect the negative value of Input Data well when it increases.
>
> As shown in the image, sigmoid and tanh can reflect negative numbers well, while the latter two rarely can.
>
> (a) Sigmoid function; (b) Tanh function; (c) ReLU function; (d) Leaky ReLU function.
>
> ![image](https://user-images.githubusercontent.com/50206883/130339004-cce06838-537f-4256-8ef0-e872b12a1b14.png)
> 
> Image from [here](https://www.researchgate.net/figure/Nonlinear-function-a-Sigmoid-function-b-Tanh-function-c-ReLU-function-d-Leaky_fig3_323617663)

>
> I set the activator as RMS Prop, which adjusts the learning rate appropriately by considering the slope and reduces the loss, and Adam, which reduces the loss by considering the direction of the step and the learning rate.
>
> According to the experiment ["A Complete Guide to Adam and RMSprop Optimizer"](https://medium.com/analytics-vidhya/a-complete-guide-to-adam-and-rmsprop-optimizer-75f4502d83be) comparing the performance and aspects of Adam and RMSprop, Adam and RMSprop are much more accurate and faster than other optimizers.
>
> ![image](https://user-images.githubusercontent.com/50206883/130339363-4a8c54ef-f899-42c9-8383-48bf492c0693.png)
>
> Image from [here](https://medium.com/analytics-vidhya/a-complete-guide-to-adam-and-rmsprop-optimizer-75f4502d83be)
>

> I set CV to 10 to have sufficiently number of Cross-Validation 
>
> I repeated the Random Search CV numerous times and found that the dropout values of the optimal parameters were small and the l1 and l2 values were very small, so I set the range as below.
>
>

> ANN
>
> ANN with Zero hidden layers.
>
> ![image](https://user-images.githubusercontent.com/50206883/130339652-299c6683-19dd-41a6-950f-fe32439cc33e.png)
>
> ANN with One hidden layers. 
>
> ![image](https://user-images.githubusercontent.com/50206883/130339714-69c6d54f-086c-4b7a-bec3-d0be515b43b0.png)
>

> DNN
>
> DNN with Two hidden layers
>
> ![image](https://user-images.githubusercontent.com/50206883/130339726-47244ed6-ecf9-487b-8ba0-f3ede7dd1b15.png)
>
> DNN with Three hidden layers
>
> ![image](https://user-images.githubusercontent.com/50206883/130339869-1333e3c8-3625-4e98-9700-49c8a0522463.png)


2. Sequential Models

> The sequence length of Input data in Sequential Models is 7. Because it is very short, the model is likely to be overfit to the train data.
>
> I therefore came to the conclusion that it is optimal to set the range of the recurrent dropout and dropout, l1 l2 values of the Hyperparameter range as shown below through repeated Random Search CV experiments.
>
> Also, I decided to set TimeSeries Split to 4 because I thought it would be difficult to reflect the overall pattern if I split it too tightly.


> Sequential Models
>
> Simple RNN
>
> ![image](https://user-images.githubusercontent.com/50206883/130340268-23559a4d-6b11-4073-859a-dfada313a4f3.png)
>
> LSTM
>
> ![image](https://user-images.githubusercontent.com/50206883/130340272-9347fa7f-74da-4dfc-b0a3-c250de75c35c.png)
>
> GRU
>
> ![image](https://user-images.githubusercontent.com/50206883/130340275-daf82f1d-1780-4764-9776-cfd518eceed4.png)


3. Sequential Model with Dense layers

> I made it basically a combination of the two parameters above.
>
> With the range set like this, I decided to observe which parameters would be optimal when the two layers were combined.

> Sequential Model with Dense layers
>
> LSTM with dense layer
>
> ![image](https://user-images.githubusercontent.com/50206883/130340340-404b6026-42cd-4df0-a333-0e2131976d19.png)
>
> GRU with dense layer
>
> ![image](https://user-images.githubusercontent.com/50206883/130340345-3bbfe6c1-95e9-44a4-8336-589c8d400a46.png)


4. 1D CNN with Sequential Models


>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
>
