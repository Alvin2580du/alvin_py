install.packages('copula')

# Ar-garch ???????????????
# ??????????????????, ?????????,0-1??????
# ????????????????????????couple??????,????????????????????? 

install.packages("rugarch")
require(rugarch)
data <- rnorm(1000)
fit <- ugarchfit(spec = spec, data = data[,1], solver.control = list(trace=0))
# Retrieve ARMA(1,1) and GARCH(1,1) coefficients:
garch@fit$coef
# Retrieve time-varying standard deviation:
garch@fit$sigma
# Retrieve standardized N(0,1) ARMA(1,1) disturbances:
garch@fit$z
# See what else you can pull out of the fit:
str(garch)

setwd("D:\\alvin_py\\business\\p201908\\5321_1000")
library(rugarch)
data<-  read.csv("hj_ry.csv")
head(data)

coefs_list <- c()

model_data <- na.omit(data$ln.P_USDJPY.)

model_data <- na.omit(data$ln.P_GOLD.)

spec <- ugarchspec(variance.model = list(model = "sGARCH", 
                                         garchOrder = c(1, 1), 
                                         submodel = NULL, 
                                         external.regressors = NULL, 
                                         variance.targeting = FALSE), 
                   
                   mean.model     = list(armaOrder = c(1, 0), 
                                         external.regressors = NULL, 
                                         distribution.model = "norm", 
                                         start.pars = list(), 
                                         fixed.pars = list()))
fit <- ugarchfit(spec = spec, data = model_data, solver.control = list(trace=0))
str(fit)

# rmgarch???,dccspec???dccfit????????????



resd <- residuals(fit)
resd_y <- pnorm(resd)
resd_y1 <- pnorm(resd)



score<-data.frame(resd_y=resd_y, resd_y1=resd_y1)

write.csv(score, "score.csv")



library(copula)
set.seed(100)
myCop <- normalCopula(param=c(0.4,0.2,-0.8), dim = 3, dispstr = "un")
myMvd <- mvdc(copula=myCop, margins=c("gamma", "beta", "t"),
              paramMargins=list(list(shape=2, scale=1),
                                list(shape1=2, shape2=2), 
                                list(df=5)))

p<-xvCopula(normalCopula(), score)
xvCopula(gumbelCopula(), x)
xvCopula(frankCopula(), x)
xvCopula(joeCopula(), x)
xvCopula(claytonCopula(), x)
xvCopula(normalCopula(), x)
xvCopula(tCopula(), x)
xvCopula(plackettCopula(), x)


# 92.293
normal.cop <- normalCopula(1, dim=2)
fit.tau <- fitCopula(normalCopula(), score, method="itau")
confint(fit.tau) # work fine !
confint(fit.tau, level = 0.98)
summary(fit.tau) # a bit more, notably "Std. Error"s
coef(fit.tau)# named vector
coef(fit.tau, SE = TRUE)# matrix
fit.rho <- fitCopula(normalCopula(), score, method="irho")
summary(fit.rho)
fit.mpl <- fitCopula(normalCopula(),score, method="mpl")
fit.mpl



#ugarchfit(spec, data, out.sample = 0, solver = "solnp", solver.control = list(),fit.control = list(stationarity = 1, fixed.se = 0, scale = 0), ...)
# ??????,spec???ugarchspec???????????????,data??????????????????solver??????????????????solver.control??????????????????,fit.control?????????????????????


variance.model = list(model = "sGARCH", garchOrder = c(1, 1),submodel = NULL, external.regressors = NULL, variance.targeting = FALSE)
mean.model = list(armaOrder = c(1, 1), include.mean = TRUE, archm = FALSE, archpow = 1, arfima = FALSE, external.regressors = NULL, archex = FALSE)
distribution.model = "norm"
distribution.model = 

myspec=ugarchspec(variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1), submodel = NULL, external.regressors = NULL, variance.targeting = FALSE),
                  mean.model = list(armaOrder = c(1, 1), include.mean = TRUE, archm = FALSE, archpow = 1, arfima = FALSE, external.regressors = NULL, archex = FALSE),
                  distribution.model = "norm")



data(sp500ret)
head(sp500ret)
myfit=ugarchfit(myspec,data=sp500ret,solver="solnp")
myfit
str(myfit)
plot(myfit)

z = residuals(myfit)/sigma(myfit)
coef(myfit)
skew = dskewness("sstd",skew = coef(myfit)["skew"], shape= coef(myfit)["shape"])
# add back 3 since dkurtosis returns the excess kurtosis
kurt = 3+dkurtosis("sstd",skew = coef(myfit)["skew"], shape= coef(myfit)["shape"])
print(GMMTest(z, lags = 1))



myfit@fit$residuals

res <- sigma(myfit)
res
# rugarch????????????????????????????????????as.data.frame???????????????????????????????????????
as.data.frame(myfit,which="fitted")

#??????????????????:
as.data.frame(myfit,which="residuals")

#??????????????????:
as.data.frame(myfit,which="sigma")

#??????,???????????????????????????:
as.data.frame(myfit,which=all)

#??????
as.data.frame(myfit)
#?????????????????????


# ????????????????????????,?????????ugarchforcast???????????????????????????:

forc <- ugarchforecast(myfit, n.ahead=20)
forc
path.sgarch = ugarchpath(myspec, n.sim=3000, n.start=1, m.sim=1)







ts.plot(model_data)




uspec = ugarchspec(mean.model = list(armaOrder = c(2,1)),
                   variance.model = list(garchOrder = c(1,1), model = "sGARCH",
                                         variance.targeting=FALSE),distribution.model = "norm")

spec1 = ugarchspec(uspec = multispec(replicate(3, uspec) ), asymmetric = TRUE,
                   distribution.model = list(copula = "mvnorm", method = "Kendall",
                                             time.varying = TRUE, transformation = "parametric"))


library("ccgarch")

nobs <- 1000; cut <- 1000
a <- c(0.003, 0.005, 0.001)
A <- diag(c(0.2,0.3,0.15))
B <- diag(c(0.75, 0.6, 0.8))
uncR <- matrix(c(1.0, 0.4, 0.3, 0.4, 1.0, 0.12, 0.3, 0.12, 1.0),3,3)
dcc.para <- c(0.01,0.98)
dcc.data <- dcc.sim(nobs, a, A, B, uncR, dcc.para, model="diagonal")
# Estimating a DCC-GARCH(1,1) model
dcc.results <- dcc.estimation(inia=a, iniA=A, iniB=B, ini.dcc=dcc.para,
                              dvar=dcc.data$eps, model="diagonal")




uspec = ugarchspec(mean.model = list(armaOrder = c(0,0)), variance.model = list(garchOrder = c(1,1), model = "sGARCH"), distribution.model = "norm")
spec1 = dccspec(uspec = multispec( replicate(8, uspec)), dccOrder = c(1,1), distribution = "mvnorm")
fit1 = dccfit(spec1, data = Dat, out.sample = 141, fit.control = list(eval.se=T))
print(fit1)

#Forecast
dcc.focast=dccforecast(fit1, n.ahead = 1, n.roll = 0) 




spec = ugarchspec(variance.model = list(model = "sGARCH"), distribution.model = "std")
cl = makePSOCKcluster(10)
cl
#????????????
roll = ugarchroll(spec, rlogdiff, n.start =300,refit.every = 300,
                  refit.window = "moving", solver = "hybrid", calculate.VaR = TRUE,
                  VaR.alpha = c(0.01, 0.025, 0.05), cluster = cl,keep.coef = TRUE)
report(roll, type = "fpm")


myspec=ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0,0), include.mean = TRUE),
  distribution.model = "std"
)

myfit=ugarchfit(myspec,data=rlogdiff,solver="gosolnp")
myfit


# ???????????????

par(mfrow=c(1,3),oma=c(0.2,0.2,0.2,0.2))
hist(rlogdiff,main="Shanghai Composite Index Log Return Distribution",col="yellow",xlab="",ylim=c(0,0.4),probability=T)
lines(density(rlogdiff),lwd=1);rug(rlogdiff)#first graph
qqnorm(rlogdiff);qqline(rlogdiff)#second graph
plot(rlogdiff,ylab="value");abline(h=0,lty=2)#third graph

# (2)???????????????
#??? ADF  p<0.0  ??????????????????????????????
adf.test(rlog,alt="stationary")#??????  ????????????????????????
adf.test(rlogdiffdata,alt="stationary")#????????????????????????


# ARCH ????????????
#??????arima????????????,???????????????LM??????
armamodel=auto.arima(rlogdiff)#????????????AIC????????????,????????????????????????
armamodel
plot(residuals(armamodel))
par(mfrow=c(1,1))
lmresult=McLeod.Li.test(y=residuals(armamodel))#??????arch???????????????

#?????????????????????
plot(myfit,which=8)
plot(myfit,which=9)
shapiro.test(coredata(residuals(myfit)))#?????????,?????????????????????,P??????????????????
#?????????????????????
acf(coredata(residuals(myfit)))
acf(residuals(myfit))
plot(myfit,which=10)
plot(myfit,which=11)
#??????????????????
myfit #???P???????????????
#????????????  ????????????
plot(myfit,which=3)
plot(residuals(myfit)) #?????????

plot(myfit,which=12)




install.packages("fGarch")
install.packages("rugarch")
install.packages("tseries")
library(fGarch)
library(rmgarch)
library(rugarch)
library(tseries)
library(zoo)
#Daten runterladen
ibm <- get.hist.quote(instrument = "DB",  start = "2005-11-21",
                      quote = "AdjClose")
sys<- get.hist.quote(instrument = "^STOXX50E",  start = "2005-11-21",
                     quote = "AdjClose")

#Returns
retibm<-diff(log(ibm))
retsys<-diff(log(sys))

# univariate normal GARCH(1,1) for each series
garch11.spec = ugarchspec(mean.model = list(armaOrder = c(0,0)), 
                          variance.model = list(garchOrder = c(1,0), 
                                                model = "sGARCH"), 
                          distribution.model = "norm")

# dcc specification - GARCH(1,1) for conditional correlations
dcc.garch11.spec = dccspec(uspec = multispec( replicate(2, garch11.spec) ), 
                           dccOrder = c(1,1), 
                           distribution = "mvnorm")
dcc.garch11.spec

MSFT.GSPC.ret = merge(retsys,retibm)
plot(MSFT.GSPC.ret)
dcc.fit <- dccfit(dcc.garch11.spec, data = na.omit(MSFT.GSPC.ret))

resd <- residuals(dcc.fit)
resd




