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

![image](https://user-images.githubusercontent.com/50206883/128275492-f229e8bc-08c7-445b-a83b-1afed48d9114.png)

### 1.2 COVID_19 articles collected daily by [Naver](https://naver.com), a large Korean portal site.

1. There are several factors affecting the number of new confirmed cases of COVID_19.

2. One is data on the number of daily confirmed cases so far, and the other is whether there is a cluster of COVID_19 cases and the transmission pattern.

3. Since I could not arbitrarily judge the latter one, I have decided to collect daily COVID_19 articles and analyze them to judge it.


![image](https://user-images.githubusercontent.com/50206883/128275695-7a13624b-fb8b-4103-9a20-dce5ba8dea7a.png)
![image](https://user-images.githubusercontent.com/50206883/128272941-55dca9e0-29b0-4a79-bc3f-55c7daaf7676.png)
![image](https://user-images.githubusercontent.com/50206883/128272963-3bbdadfd-7315-4fb1-a819-77fe6dea4d15.png)


### 1.3 Processed data

1. 
