setwd("????????????")
install.packages("fGarch")
install.packages("rugarch")
install.packages("rmgarch")
install.packages("tseries")
install.packages("copula")

library(fGarch)
library(rmgarch)
library(rugarch)
library(tseries)
library(zoo)
library(copula)

data<-  read.csv("hj_ry.csv")
head(data)
data <- na.omit(data)
head(data)

retsyspobs <- pobs(data$ln.P_USDJPY., ties.method = "max")
retibmpobs <- pobs(data$ln.P_GOLD., ties.method = "max")
plot(retsyspobs,retibmpobs)
#??????
MSFT.GSPC.ret1 = data.frame(retsys=retsyspobs,retibm=retibmpobs)
row.names(MSFT.GSPC.ret1) = data$Date


cor(retsys, retibm)
# ????????????
cor.spearman<-cor(MSFT.GSPC.ret1, method = "spearman")
cor.pearson<-cor(MSFT.GSPC.ret1, method = "pearson")
cor.kendall<-cor(MSFT.GSPC.ret1, method = "kendall")
cor.spearman
cor.pearson
cor.kendall

head(MSFT.GSPC.ret1[,1,drop=FALSE])

# ????????? AR-GARCH(1,1)
garch11.spec <- ugarchspec(mean.model = list(armaOrder = c(1,0)), 
                          variance.model = list(garchOrder = c(1,1), 
                                                model = "sGARCH"), 
                          distribution.model = "norm")

garch.fit <- ugarchfit(garch11.spec,data=MSFT.GSPC.ret1[,1,drop=FALSE],solver="solnp")
plot(garch.fit)
garch.resd <- residuals(garch.fit)
garch.resdpobs <- pobs(garch.resd)
garch.resdpobs

garch.pnorm_resd <- pnorm(garch.resd)
garch.pnorm_resd <- as.matrix(garch.pnorm_resd)

# ======================================

garch11.spec.retibm <- ugarchspec(mean.model = list(armaOrder = c(1,0)), 
                           variance.model = list(garchOrder = c(1,1), 
                                                 model = "sGARCH"), 
                           distribution.model = "norm")

garch.fit.retibm <- ugarchfit(garch11.spec.retibm,data=MSFT.GSPC.ret1[,2,drop=FALSE],solver="solnp")
plot(garch.fit.retibm)
garch.resd.retibm <- residuals(garch.fit.retibm)
garch.resdpobs.retibm <- pobs(garch.resd.retibm)
garch.resdpobs.retibm

garch.pnorm_resd.retibm <- pnorm(garch.resd.retibm)
garch.pnorm_resd.retibm <- as.matrix(garch.pnorm_resd.retibm)

garch.data.frame = data.frame(x1=garch.pnorm_resd.retibm, x2=garch.pnorm_resd)
garch.data.frame = as.matrix(garch.data.frame)
class(garch.data.frame)

# ?????????,dcc- GARCH(1,1)
dcc.garch11.spec = dccspec(uspec = multispec(replicate(2, garch11.spec)), 
                           dccOrder = c(1,1),  distribution = "mvnorm")

dcc.fit <- dccfit(dcc.garch11.spec, data = MSFT.GSPC.ret1, solver="solnp")
str(dcc.fit)
plot(dcc.fit)

dcc.resd <- residuals(dcc.fit)
dcc.resd
dcc.resdpobs <- pobs(dcc.resd)
dcc.resdpobs
dcc.pnorm_resd <- pnorm(dcc.resd)
head(dcc.pnorm_resd)
dcc.pnorm_resd <- as.matrix(dcc.pnorm_resd)
class(dcc.pnorm_resd)

# ??????copula??????-?????????
library(copula)
library(psych)
# =============== Copula ml ======================  ????????????
n.cop <- normalCopula(dim=2)
clayton <- claytonCopula(dim = 2, param = 19)
t.cop <- tCopula(dim = 2)
fit.ml.1 <- fitCopula(n.cop, garch.data.frame, method="ml")
fit.ml.1
fit.ml.2 <- fitCopula(t.cop, garch.data.frame, method="ml")
fit.ml.2


# ===============dcc normalCopula======================
n.cop <- normalCopula(dim=2)
# ml,mpl,itau,irho
fit.tau <- fitCopula(n.cop, data=dcc.pnorm_resd, method="itau")
fit.tau
confint(fit.tau, level = 0.98)
summary(fit.tau) # a bit more, notably "Std. Error"s
coef(fit.tau, SE = TRUE)# matrix
xvCopula(n.cop, pobs(dcc.pnorm_resd), method="itau")
# [1] 808.9238

# =================dcc  claytonCopula====================
clayton <- claytonCopula(dim = 2, param = 19)
fit.tau <- fitCopula(clayton, data=dcc.pnorm_resd, method="irho")
confint(fit.tau, level = 0.98)
summary(fit.tau) # a bit more, notably "Std. Error"s
coef(fit.tau)# named vector
coef(fit.tau, SE = TRUE)# matrix
xvCopula(clayton, dcc.pnorm_resd, method="itau")
# [1] -375.1408

# ==================dcc  tCopula===================
t.cop <- tCopula(dim = 2)
fit.ml.t = fitCopula(copula=t.cop, data=dcc.pnorm_resd, method="itau", start=c(omega, 10)) 
fit.ml.t@estimate
summary(fit.ml.t)
confint(fit.ml.t, level = 0.98)
coef(fit.ml.t)# named vector
coef(fit.ml.t, SE = TRUE)# matrix
xvCopula(t.cop, dcc.pnorm_resd, method="itau")

