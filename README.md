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
>
>
>
>
>
