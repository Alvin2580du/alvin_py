#?????????
install.packages("zoo")
install.packages("forecast")
install.packages("tseries")
install.packages("e1071")
# ?????????
library("e1071")
library("tseries")
library("zoo")
library("forecast")
setwd('D:\\alvin_py\\business\\p201909\\timeseriesPredict300')
dt <- read.csv("123456789.csv") #????????????

airline2 <- dt[1:68,3]  # ??????????????????

airts <- ts(airline2,start=c(6,1),frequency=30)
plot.ts(airts) # ?????????????????????,???2011???2????????????,12????????????

airdiff <- diff(diff(airts, differences=1))
plot.ts(airdiff) #???????????????,??????????????????diff???????????????,??????????????????;?????????diff???????????????????????????
adf.test(airdiff, alternative="stationary", k=0) 
#????????????????????? ?????????????????????????????????acf and pcaf  ?????????????????????

#adf???pacf?????????,????????????
acf(airdiff, lag.max=20)
acf(airdiff, lag.max=20,plot=FALSE)
pacf(airdiff, lag.max=20)
pacf(airdiff, lag.max=20,plot=FALSE)

auto.arima(airdiff,trace=T)  #????????????

#???????????????????????????,?????????????????????
airarima1 <- arima(airdiff, order=c(1,0,0), seasonal=list(order=c(1,0,0),period=NA),  method="ML")
airarima1
  
airforecast <- forecast(airarima1,h=16,level=c(99.5))
airforecast

plot(airforecast)   # ????????????
write.csv(airforecast,file="yc1.csv") #????????????

sub1=as.matrix(airforecast$residuals)  #???????????????
sub1
#4???????????????
xindt=matrix(NA,4,length(sub1)-6)
for (i in 1: 10){
  xindt[,i]=c(sub1[i],sub1[i+1],sub1[i+2],sub1[i+3])
}

inputData<-cbind(t(xindt[,1:9]),sub1[5:13])

set.seed(100) # for reproducing results
rowIndices <- 1 : nrow(inputData) # prepare row indices
sampleSize <- 0.8 * length(rowIndices) # training sample size
trainingRows <- sample (rowIndices, sampleSize) # random sampling
trainingData <- inputData[trainingRows, ] # training data
testData <- inputData[-trainingRows, ] # test data

svmfit <- svm (V5 ~ ., data = trainingData)

print(svmfit)
predict(svmfit, trainingData[,1:4])
compareTable <- cbind(testData[,5], predict(svmfit, testData[,1:4]))  # comparison table

yc1=predict(svmfit, t(xindt))
write.csv(yc1,file="yc2.csv")#????????????


#???????????????1???
h1=sub1[10:13,1]
h1=as.matrix(h1)
row.names(h1)<-c("V1","V2","V3","V4")
th1=predict(svmfit, t(h1))
th1

#?????????????????????
h2=c(sub1[11:13,1],th1)
h2=as.matrix(h2)
row.names(h2)<-c("V1","V2","V3","V4")
th2=predict(svmfit, t(h2))
th2

#????????????????????????
h3=c(sub1[12:13,1],th1,th2)
h3=as.matrix(h3)
row.names(h3)<-c("V1","V2","V3","V4")
th3=predict(svmfit, t(h3))
th3
