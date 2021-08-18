### README.md

 - Hello!
 - This is an experimental project to compare COVID-19 daily/weekly confirmed patient prediction accuracy using various deep learning models.
 - Things will be updated frequently.






# 1. Data


#### My data are divided into two main categories. The number of confirmed COVID_19 patients and crawled data from news articles every day in Korea.





### 1.1 The number of confirmed COVID_19 patients.

1. This Data consists of the number of confirmed COVID_19 patients everyday in South Korea. 

2. All data were obtained from the [Korea Data Exchange](https://kdx.kr/main).

3. From February 21, 2020, when the epidemic began to spread in earnest, data was collected and obtained until August 3, 2021.

4. I've transformed this data into the number of new confirmed cases per day, as there were only cumulative values of Korean COVID_19 cases in it.

> ![image](https://user-images.githubusercontent.com/50206883/128275492-f229e8bc-08c7-445b-a83b-1afed48d9114.png)





### 1.2 COVID_19 articles collected daily by [Naver](https://naver.com), a large Korean portal site.

1. There are several factors affecting the number of new confirmed cases of COVID_19.

2. One is data on the number of daily confirmed cases so far, and the other is whether there is a cluster of COVID_19 cases and the transmission pattern.

3. Since I could not arbitrarily judge the latter one, I have decided to collect daily COVID_19 articles and analyze them to judge it.


> ![image](https://user-images.githubusercontent.com/50206883/128275695-7a13624b-fb8b-4103-9a20-dce5ba8dea7a.png)
> ![image](https://user-images.githubusercontent.com/50206883/128272941-55dca9e0-29b0-4a79-bc3f-55c7daaf7676.png)
> ![image](https://user-images.githubusercontent.com/50206883/128272963-3bbdadfd-7315-4fb1-a819-77fe6dea4d15.png)





### 1.3 Processed data



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




