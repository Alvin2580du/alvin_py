## By Marius Hofert, Ivan Kojadinovic, Martin Maechler, Jun Yan

## R script for Chapter 2 of Elements of Copula Modeling with R


### 2.1 Definition and characterization ########################################

### Independence copula

library(copula)
d <- 2
ic <- indepCopula(dim = d)
ic

set.seed(2008)
u <- runif(d) # a random point in the unit hypercube
(Pi <- pCopula(u, copula = ic)) # the value of the independence copula at u

stopifnot(all.equal(Pi, prod(u))) # check numerical equality of the samples

wireframe2  (ic, FUN = pCopula, # surface plot of the independence copula
             col.4 = adjustcolor("black", alpha.f = 0.25))
contourplot2(ic, FUN = pCopula) # contour plot of the independence copula


### C-volumes

a <- c(1/4, 1/2) # lower left end point
b <- c(1/3, 1) # upper right end point
stopifnot(0 <= a, a <= 1, 0 <= b, b <= 1, a <= b) # check
p <- (b[1] - a[1]) * (b[2] - a[2]) # manual computation
stopifnot(all.equal(prob(ic, l = a, u = b), p)) # check

n <- 1000 # sample size
set.seed(271) # set a seed (for reproducibility)
U <- rCopula(n, copula = ic) # generate a sample of the independence copula
plot(U, xlab = quote(U[1]), ylab = quote(U[2]))

set.seed(271)
stopifnot(all.equal(U, matrix(runif(n * d), nrow = n)))

set.seed(314)
U <- rCopula(1e6, copula = ic) # large sample size for good approximation
## Approximate the Pi-volume by the aforementioned proportion
p.sim <- mean(a[1] < U[,1] & U[,1] <= b[1] & a[2] < U[,2] & U[,2] <= b[2])
stopifnot(all.equal(p.sim, p, tol = 1e-2)) # note: may depend on seed


### Frank copula

d <- 2 # dimension
theta <- -9 # copula parameter
fc <- frankCopula(theta, dim = d) # define a Frank copula

set.seed(2010)
n <- 5 # number of evaluation points
u <- matrix(runif(n * d), nrow = n) # n random points in [0,1]^d
pCopula(u, copula = fc) # copula values at u

dCopula(u, copula = fc) # density values at u

wireframe2(fc, FUN = pCopula, # wireframe plot (copula)
           draw.4.pCoplines = FALSE)
wireframe2(fc, FUN = dCopula, delta = 0.001) # wireframe plot (density)
contourplot2(fc, FUN = pCopula) # contour plot (copula)
contourplot2(fc, FUN = dCopula, n.grid = 72, # contour plot (density)
             lwd = 1/2)

set.seed(1946)
n <- 1000
U  <- rCopula(n, copula = fc)
U0 <- rCopula(n, copula = setTheta(fc, value = 0))
U9 <- rCopula(n, copula = setTheta(fc, value = 9))
plot(U,  xlab = quote(U[1]), ylab = quote(U[2]))
plot(U0, xlab = quote(U[1]), ylab = quote(U[2]))
plot(U9, xlab = quote(U[1]), ylab = quote(U[2]))


### Clayton copula

d <- 3
cc <- claytonCopula(4, dim = d) # theta = 4

set.seed(2013)
n <- 5
u <- matrix(runif(n * d), nrow = n) # random points in the unit hypercube
pCopula(u, copula = cc) # copula values at u
dCopula(u, copula = cc) # density values at u

set.seed(271)
U <- rCopula(1000, copula = cc)
splom2(U, cex = 0.3, col.mat = "black")


### Gumbel-Hougaard copula

gc <- gumbelCopula(3) # theta = 3 (note the default dim = 2)

set.seed(1993)
U <- rCopula(1000, copula = gc)
plot(U, xlab = quote(U[1]), ylab = quote(U[2]))
wireframe2(gc, dCopula, delta = 0.025) # wireframe plot (density)


### 2.2 The Frechet-Hoeffding bounds ###########################################

### Frechet-Hoeffding bounds

set.seed(1980)
U <- runif(100)
plot(cbind(U, 1-U), xlab = quote(U[1]), ylab = quote(U[2]))
plot(cbind(U, U),   xlab = quote(U[1]), ylab = quote(U[2]))

u <- seq(0, 1, length.out = 40) # subdivision points in each dimension
u12 <- expand.grid("u[1]" = u, "u[2]" = u) # build a grid
W <- pmax(u12[,1] + u12[,2] - 1, 0) # values of W on grid
M <- pmin(u12[,1], u12[,2]) # values of M on grid
val.W <- cbind(u12, "W(u[1],u[2])" = W) # append grid
val.M <- cbind(u12, "M(u[1],u[2])" = M) # append grid
wireframe2(val.W)
wireframe2(val.M)
contourplot2(val.W, xlim = 0:1, ylim = 0:1)
contourplot2(val.M, xlim = 0:1, ylim = 0:1)


### Marshall-Olkin copulas

## A Marshall-Olkin copula
C <- function(u, alpha)
  pmin(u[,1] * u[,2]^(1 - alpha[2]), u[,1]^(1 - alpha[1]) * u[,2])
alpha <- c(0.2, 0.8)
val <- cbind(u12, "C(u[1],u[2])" = C(u12, alpha = alpha)) # append C values
## Generate data
set.seed(712)
V <- matrix(runif(1000 * 3), ncol = 3)
U <- cbind(pmax(V[,1]^(1/(1 - alpha[1])), V[,3]^(1/alpha[1])),
           pmax(V[,2]^(1/(1 - alpha[2])), V[,3]^(1/alpha[2])))
## Plots
wireframe2(val)
plot(U, xlab = quote(U[1]), ylab = quote(U[2]))


### 2.3 Sklar's Theorem ########################################################

### First part of Sklar's Theorem - decomposition

library(mvtnorm)
d <- 2 # dimension
rho <- 0.7 # off-diagonal entry of the correlation matrix P
P <- matrix(rho, nrow = d, ncol = d) # build the correlation matrix P
diag(P) <- 1
set.seed(64)
u <- runif(d) # generate a random evaluation point
x <- qnorm(u)
pmvnorm(upper = x, corr = P) # evaluate the copula C at u

nc <- normalCopula(rho) # normal copula (note the default dim = 2)
pCopula(u, copula = nc) # value of the copula at u

nu <- 3 # degrees of freedom
x. <- qt(u, df = nu)
pmvt(upper = x., corr = P, df = nu) # evaluate the t copula at u

try(pmvt(upper = x., corr = P, df = 3.5))

tc <- tCopula(rho, dim = d, df = nu)
pCopula(u, copula = tc) # value of the copula at u


### Second part of Sklar's Theorem - composition

H.obj <- mvdc(claytonCopula(1), margins = c("norm", "exp"),
              paramMargins = list(list(mean = 1, sd = 2), list(rate = 3)))

set.seed(1979)
z <- cbind(rnorm(5, mean = 1, sd = 2), rexp(5, rate = 3)) # evaluation points
pMvdc(z, mvdc = H.obj) # values of the df at z

dMvdc(z, mvdc = H.obj) # values of the corresponding density at z

set.seed(1975)
X <- rMvdc(1000, mvdc = H.obj)

plot(X, cex = 0.5, xlab = quote(X[1]), ylab = quote(X[2]))
contourplot2(H.obj, FUN = dMvdc, xlim = range(X[,1]), ylim = range(X[,2]),
             n.grid = 257)

library(nor1mix)
## Define and visualize two mixtures of normals
plot(nm1 <- norMix(c(1, -1), sigma = c( .5, 1), w = c(.2, .8)))
plot(nm2 <- norMix(c(0,  2), sigma = c(1.5, 1), w = c(.3, .7)))

H.obj.m <- mvdc(claytonCopula(1), margins = c("norMix", "norMix"),
                paramMargins = list(nm1, nm2))

set.seed(271)
X <- rMvdc(1000, mvdc = H.obj.m)

plot(X, cex = 0.5, xlab = quote(X[1]), ylab = quote(X[2]))
contourplot2(H.obj.m, FUN = dMvdc, xlim = range(X[,1]), ylim = range(X[,2]),
             n.grid = 129)


### Risk aggregation

## Define parameters of the three margins
th <- 2.5 # Pareto parameter
m <- 10 # mean of the lognormal
v <- 20 # variance of the lognormal
s <- 4 # shape of the gamma underlying the loggamma
r <- 5 # rate of the gamma underlying the loggamma
## Define list of marginal dfs
qF <- list(qPar = function(p) (1 - p)^(-1/th) - 1,
           qLN  = function(p) qlnorm(p, meanlog = log(m)-log(1+v/m^2)/2,
                                     sdlog = sqrt(log(1+v/m^2))),
           qLG  = function(p) exp(qgamma(p, shape = s, rate = r)))
## Generate the data
set.seed(271) # for reproducibility
X <- sapply(qF, function(mqf) mqf(runif(2500))) # (2500, 3)-matrix

##' @title Nonparametric VaR estimate under a t copula
##' @param X loss matrix
##' @param alpha confidence level(s)
##' @param rho correlation parameter of the t copula
##' @param df degrees of freedom parameter of the t copula
##' @return Nonparametric VaR estimate under the t copula (numeric)
VaR <- function(X, alpha, rho, df = 3.5)
{
  stopifnot(is.matrix(X), 0 <= rho, rho <= 1, length(rho) == 1,
            0 < alpha, alpha < 1, length(alpha) >= 1)
  n <- nrow(X) # sample size
  d <- ncol(X) # dimension
  ## Simulate from a t copula with d.o.f. parameter 3.5 and exchangeable
  ## correlation matrix with off-diagonal entry rho. Also compute the
  ## componentwise ranks.
  ## Note: We can set the seed here as we can estimate VaR for all
  ##       confidence levels based on the same copula sample. We
  ##       even *should* set the seed here to minimize the variance
  ##       of the estimator and make the results more comparable.
  set.seed(271)
  U <- rCopula(n, copula = tCopula(rho, dim = d, df = df))
  rk <- apply(U, 2, rank)
  ## Componentwise reorder the data according to these ranks to
  ## mimic the corresponding t copula dependence among the losses
  Y <- sapply(1:d, function(j) sort(X[,j])[rk[,j]])
  ## Build row sums to mimic a sample from the distribution of the
  ## sum under the corresponding t copula.
  S <- rowSums(Y)
  ## Nonparametrically estimate VaR for all confidence levels alpha
  ## Note: We use the mathematical definition ('type = 1') of a
  ##       quantile function here
  quantile(S, probs = alpha, type = 1, names = FALSE)
}

alpha <- c(0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
           0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999) # confidence levels
rho <- seq(0, 1, by = 0.1) # parameter of the homogeneous t copula
grid <- expand.grid("alpha" = alpha, "rho" = rho)[,2:1] # build a grid
VaR.fit <- sapply(rho, function(r)
  VaR(X, alpha = alpha, rho = r)) # (alpha, rho)
res <- cbind(grid, "VaR[alpha](L^'+')" = as.vector(VaR.fit))

wireframe2(res)
library(qrmtools)
worst.VaR <- sapply(alpha, function(a) mean(ARA(a, qF = qF)$bounds))
plot(alpha, worst.VaR, type = "b", col = 2,
     xlab = quote(alpha), ylab = quote(VaR[alpha](L^'+')),
     ylim = range(VaR.fit, worst.VaR)) # computed with the ARA
lines(alpha, apply(VaR.fit, 1, max), type = "b", col = 1) # simulated
legend("topleft", bty = "n", lty = rep(1, 2), col = 2:1,
       legend = c(expression("Worst"~VaR[alpha]~"according to ARA()"),
                  expression("Worst"~VaR[alpha]~"under"~t[3.5]~"copulas")))

## Computing worst VaR in the three-dimensional case
wVaR <- ARA(0.99, qF = qF) # compute worst VaR (bounds)
X <- wVaR[["X.rearranged"]]$up # extract rearranged matrix (upper bound)
U <- pobs(X) # compute pseudo-observations
pairs2(U) # approx. sample of a copula leading to worst VaR for our marg. dfs
## Computing worst VaR in the bivariate case
wVaR. <- ARA(0.99, qF = qF[1:2]) # compute worst VaR (bounds)
X. <- wVaR.[["X.rearranged"]]$up # extract rearranged matrix (upper bound)
U. <- pobs(X.) # compute pseudo-observations
plot(U., xlab = quote(U[1]), ylab = quote(U[2]))


### 2.4 The invariance principle ###############################################

### Sampling from a normal or t copula

n <- 1000 # sample size
d <- 2 # dimension
rho <- 0.7 # off-diagonal entry in the correlation matrix P
P <- matrix(rho, nrow = d, ncol = d) # build the correlation matrix P
diag(P) <- 1
nu <- 3.5 # degrees of freedom
set.seed(271)
X <- rmvt(n, sigma = P, df = nu) # n ind. multivariate t observations
U <- pt(X, df = nu) # n ind. realizations from the corresponding copula

set.seed(271)
U. <- rCopula(n, tCopula(rho, dim = d, df = nu))
stopifnot(all.equal(U, U.)) # test of (numerical) equality

plot(U., xlab = quote(U[1]), ylab = quote(U[2]))
plot(U,  xlab = quote(U[1]), ylab = quote(U[2]))


### From a multivariate t distribution to a t copula to a meta-t model

## Plot function highlighting three points
plotABC <- function(x, ind3, col = adjustcolor("black", 1/2), pch = 19, ...)
{
  cols <- adjustcolor(c("red","blue","magenta"), offset = -c(1,1,1,1.5)/4)
  par(pty = "s")
  plot(x, col = col, asp = 1,...)
  xy <- x[ind3, , drop = FALSE]
  points(xy, pch = pch, col = cols)
  text(xy, label = names(ind3), adj = c(0.5, -0.6), col = cols, font = 2)
}
ind3 <- c(A = 725, B = 351, C = 734) # found via 'plot(X); identify(X)'
## Scatter plot of observations from the multivariate t distribution
plotABC(X, ind3 = ind3, xlab = quote(X[1]), ylab = quote(X[2]))
## Scatter plot of observations from the corresponding t copula
plotABC(U, ind3 = ind3, xlab = quote(U[1]), ylab = quote(U[2]))
## Scatter plot of observations from the meta-t distribution
Y <- qnorm(U) # transform U (t copula) to normal margins
plotABC(Y, ind3 = ind3, xlab = quote(Y[1]), ylab = quote(Y[2]))


### Verifying the invariance principle

rho <- 0.6
P <- matrix(c(1, rho, rho, 1), ncol = 2) # the correlation matrix
C <- function(u) pCopula(u, copula = normalCopula(rho)) # normal copula
Htilde <- function(x)
  apply(cbind(log(x[,1]), -log((1-x[,2])/x[,2])), 1, function(x.)
    pmvnorm(upper = x., corr = P))
qF1tilde <- function(u) exp(qnorm(u))
qF2tilde <- function(u) 1/(1+exp(-qnorm(u)))
Ctilde <- function(u) Htilde(cbind(qF1tilde(u[,1]), qF2tilde(u[,2])))
set.seed(31)
u <- matrix(runif(5 * 2), ncol = 2) # 5 random evaluation points
stopifnot(all.equal(Ctilde(u), C(u)))

set.seed(721)
X <- rmvnorm(1000, mean = c(0,0), sigma = P) # sample from N(0, P)
## 'Sample' the copula of X directly
U <- pnorm(X)
## Transform the sample X componentwise
TX <- cbind(exp(X[,1]), plogis(X[,2])) # note: plogis(x) = 1/(1+exp(-x))
## Apply the marginal dfs to get a sample from the copula of TX
## Note: qlogis(p) == logit(p) == log(p/(1-p))
V <- cbind(pnorm(log(TX[,1])), pnorm(qlogis(TX[,2])))
stopifnot(all.equal(V, U)) # => the samples of the two copulas are the same


### 2.5 Survival copulas and copula symmetries #################################

### Survival copulas

cc <- claytonCopula(2)
set.seed(271)
U <- rCopula(1000, copula = cc) # sample from the Clayton copula
V <- 1 - U # sample from the survival Clayton copula
plot(U, xlab = quote(U[1]), ylab = quote(U[2])) # scatter plot
plot(V, xlab = quote(V[1]), ylab = quote(V[2])) # for the survival copula

wireframe2(cc,            FUN = dCopula, delta = 0.025)
wireframe2(rotCopula(cc), FUN = dCopula, delta = 0.025)


### Visually assessing radial symmetry and exchangeability

contourplot2(tCopula(0.7, df = 3.5), FUN = dCopula, n.grid = 64, lwd = 1/2)
contourplot2(gumbelCopula(2),        FUN = dCopula, n.grid = 64, lwd = 1/4,
             pretty = FALSE, cuts = 42,
             col.regions = gray(seq(0.5, 1, length.out = 128)))


### 2.6 Measures of association ################################################

### 2.6.1 Fallacies related to the correlation coefficient #####################

### Counterexample to Fallacies 3 and 4

## Evaluate the density of C for h_1(u) = 2*u*(u-1/2)*(u-1),
## h_2(u) = theta*u*(1-u) and two different thetas
u <- seq(0, 1, length.out = 20) # subdivision points in each dimension
u12 <- expand.grid("u[1]" = u, "u[2]" = u) # build a grid
dC <- function(u, th) 1 + th * (6 * u[,1] * (u[,1]-1) + 1) * (1 - 2*u[,2])
wireframe2(cbind(u12, "c(u[1],u[2])" = dC(u12, th = -1)))
wireframe2(cbind(u12, "c(u[1],u[2])" = dC(u12, th =  1)))


### Uncorrelatedness versus independence

n <- 1000
set.seed(314)
Z <- rnorm(n)
U <- runif(n)
V <- rep(1, n)
V[U < 1/2] <- -1 # => V in {-1,1}, each with probability 1/2
X <- cbind(Z, Z*V) # (X_1,X_2)
stopifnot(cor.test(X[,1], X[,2])$p.value >= 0.05) # H0:`cor=0' not rejected
Y <- matrix(rnorm(n * 2), ncol = 2) # independent N(0,1)
## Plots
plot(X, xlab = quote(X[1]), ylab = quote(X[2]))
plot(Y, xlab = quote(Y[1]), ylab = quote(Y[2]))


### Counterexample to Fallacy 5

## Function to compute the correlation bounds for LN(0, sigma_.^2) margins
corBoundLN <- function(s, bound = c("max", "min"))
{
  ## s = (sigma_1, sigma_2)
  if(!is.matrix(s)) s <- rbind(s)
  bound <- match.arg(bound)
  if(bound == "min") s[,2] <- -s[,2]
  (exp((s[,1]+s[,2])^2/2)-exp((s[,1]^2+s[,2]^2)/2)) /
    sqrt(expm1(s[,1]^2)*exp(s[,1]^2)*expm1(s[,2]^2)*exp(s[,2]^2))
}
## Evaluate correlation bounds on a grid
s <- seq(0.01, 5, length.out = 20) # subdivision points in each dimension
s12 <- expand.grid("sigma[1]" = s, "sigma[2]" = s) # build a grid
## Plots
wireframe2(cbind(s12, `underline(Cor)(sigma[1],sigma[2])` =
                   corBoundLN(s12, bound = "min")))
wireframe2(cbind(s12, `bar(Cor)(sigma[1],sigma[2])` = corBoundLN(s12)))


### 2.6.2 Rank correlation measures ############################################

### rho(), iRho(), tau() and iTau()

theta <- -0.7
stopifnot(all.equal(rho(normalCopula(theta)), 6 / pi * asin(theta / 2)))
stopifnot(all.equal(tau(normalCopula(theta)), 2 / pi * asin(theta)))
theta <- 2
stopifnot(all.equal(tau(claytonCopula(theta)), theta / (theta + 2)))
stopifnot(all.equal(tau(gumbelCopula(theta)), 1 - 1 / theta))

theta <- (0:8)/16
stopifnot(all.equal(iRho(normalCopula(), rho = 6/pi * asin(theta/2)), theta))
stopifnot(all.equal(iTau(normalCopula(), tau = 2/pi * asin(theta)),   theta))
theta <- 1:20
stopifnot(all.equal(iTau(claytonCopula(), theta / (theta + 2)), theta))
stopifnot(all.equal(iTau(gumbelCopula(),  1 - 1 / theta),       theta))

theta <- 3
iRho(claytonCopula(), rho = rho(claytonCopula(theta)))


### Estimating Spearman's rho and Kendall's tau

theta <- iRho(claytonCopula(), rho = 0.6) # true Spearman's rho = 0.6
set.seed(974)
U <- rCopula(1000, copula = claytonCopula(theta))
rho.def <- cor(apply(U, 2, rank))[1,2]      # Spearman's rho manually
rho.R   <- cor(U, method = "spearman")[1,2] # Spearman's rho from R
stopifnot(all.equal(rho.def, rho.R)) # the same
rho.R  # indeed close to 0.6

theta <- iTau(normalCopula(), tau = -0.5) # true Kendall's tau = -0.5
set.seed(974)
U <- rCopula(1000, copula = normalCopula(theta))
p.n <- 0
for(i in 1:(n-1)) # number of concordant pairs (obviously inefficient)
  for(j in (i+1):n)
    if(prod(apply(U[c(i,j),], 2, diff)) > 0) p.n <- p.n + 1
tau.def <- 4 * p.n / (n * (n - 1)) - 1   # Kendall's tau manually
tau.R <- cor(U, method = "kendall")[1,2] # Kendall's tau from R
stopifnot(all.equal(tau.def, tau.R)) # the same
tau.R # close to -0.5


### Spearman's rho and Kendall's tau under counter- and comonotonicity

set.seed(75)
X <- rnorm(100)
Y <- -X^3 # perfect negative dependence
rho.counter <- cor(X, Y, method = "spearman")
tau.counter <- cor(X, Y, method = "kendall")
stopifnot(rho.counter == -1, tau.counter == -1)
Z <- exp(X) # perfect positive dependence
rho.co <- cor(X, Z, method = "spearman")
tau.co <- cor(X, Z, method = "kendall")
stopifnot(rho.co == 1, tau.co == 1)


### Spearman's rho and Kendall's tau for normal copulas

rho <- seq(-1, 1, by = 0.01) # correlation parameters of normal copulas
rho.s <- (6/pi) * asin(rho/2) # corresponding Spearman's rho
tau <- (2/pi) * asin(rho) # corresponding Kendall's tau
plot(rho, rho.s, type = "l", col = 2, lwd = 2,
     xlab = expression("Correlation parameter"~rho~"of"~C[rho]^n),
     ylab = expression("Corresponding"~rho[s]~"and"~tau))
abline(a = 0, b = 1, col = 1, lty = 2, lwd = 2)
lines(rho, tau, col = 3, lwd = 2)
legend("bottomright", bty = "n", col = 1:3, lty = c(2, 1, 1), lwd = 2,
       legend = c("Diagonal", expression(rho[s]), expression(tau)))
plot(rho, rho.s - rho, type = "l", yaxt = "n", lwd = 2,
     xlab = expression(rho), ylab = expression(rho[s]-rho))
mdiff <- max(rho.s - rho)
abline(h = c(-1, 1) * mdiff, lty = 2, lwd = 2)
rmdiff <- round(mdiff, 4)
axis(2, at = c(-mdiff, -0.01, 0, 0.01, mdiff),
     labels = as.character(c(-rmdiff, -0.01, 0, 0.01, rmdiff)))


### 2.6.3 Tail dependence coefficients #########################################

### Four distributions with N(0,1) margins and a Kendall's tau of 0.7

## Kendall's tau and corresponding copula parameters
tau <- 0.7
th.n <- iTau(normalCopula(),  tau = tau)
th.t <- iTau(tCopula(df = 3), tau = tau)
th.c <- iTau(claytonCopula(), tau = tau)
th.g <- iTau(gumbelCopula(),  tau = tau)
## Samples from the corresponding 'mvdc' objects
set.seed(271)
n <- 10000
N01m <- list(list(mean = 0, sd = 1), list(mean = 0, sd = 1)) # margins
X.n <- rMvdc(n, mvdc = mvdc(normalCopula(th.n),    c("norm", "norm"), N01m))
X.t <- rMvdc(n, mvdc = mvdc(tCopula(th.t, df = 3), c("norm", "norm"), N01m))
X.c <- rMvdc(n, mvdc = mvdc(claytonCopula(th.c),   c("norm", "norm"), N01m))
X.g <- rMvdc(n, mvdc = mvdc(gumbelCopula(th.g),    c("norm", "norm"), N01m))
##' @title Function for producing one scatter plot
##' @param X data
##' @param qu (lower and upper) quantiles to consider
##' @param lim (x- and y-axis) limits
##' @param ... additional arguments passed to the underlying plot functions
##' @return invisible()
plotCorners <- function(X, qu, lim, smooth = FALSE, ...)
{
  plot(X, xlim = lim, ylim = lim, xlab = quote(X[1]), ylab = quote(X[2]),
       col = adjustcolor("black", 0.5), ...) # or pch = 16
  abline(h = qu, v = qu, lty = 2, col = adjustcolor("black", 0.6))
  ll <- sum(apply(X <= qu[1], 1, all)) * 100 / n
  ur <- sum(apply(X >= qu[2], 1, all)) * 100 / n
  mtext(sprintf("Lower left: %.2f%%, upper right: %.2f%%", ll, ur),
        cex = 0.9, side = 1, line = -1.5)
  invisible()
}
## Plots
a. <- 0.005
q <- qnorm(c(a., 1 - a.)) # a- and (1-a)-quantiles of N(0,1)
lim <- range(q, X.n, X.t, X.c, X.g)
lim <- c(floor(lim[1]), ceiling(lim[2]))
plotCorners(X.n, qu = q, lim = lim, cex = 0.4)
plotCorners(X.t, qu = q, lim = lim, cex = 0.4)
plotCorners(X.c, qu = q, lim = lim, cex = 0.4)
plotCorners(X.g, qu = q, lim = lim, cex = 0.4)


### Computing the coefficients of tail dependence

## Clayton copula
theta <- 3
lam.c <- lambda(claytonCopula(theta))
stopifnot(all.equal(lam.c[["lower"]], 2^(-1/theta)),
          all.equal(lam.c[["upper"]], 0))
## Gumbel--Hougaard copula
lam.g <- lambda(gumbelCopula(theta))
stopifnot(all.equal(lam.g[["lower"]], 0),
          all.equal(lam.g[["upper"]], 2-2^(1/theta)))
## Normal copula
rho <- 0.7
nu <- 3
lam.n <- lambda(normalCopula(rho))
stopifnot(all.equal(lam.n[["lower"]], 0),
          all.equal(lam.n[["lower"]], lam.n[["upper"]]))
## t copula
lam.t <- lambda(tCopula(rho, df = nu))
stopifnot(all.equal(lam.t[["lower"]],
                    2*pt(-sqrt((nu+1)*(1-rho)/(1+rho)), df = nu + 1)),
          all.equal(lam.t[["lower"]], lam.t[["upper"]]))


### Tail dependence of t copulas

## Coefficient of tail dependence as a function of rho
rho <- seq(-1, 1, by = 0.01)
nu <- c(3, 4, 8, Inf)
n.nu <- length(nu)
lam.rho <- sapply(nu, function(nu.) # (rho, nu) matrix
  sapply(rho, function(rho.) lambda(tCopula(rho., df = nu.))[["lower"]]))
expr.rho <- as.expression(lapply(1:n.nu, function(j)
  bquote(nu == .(if(nu[j] == Inf) quote(infinity) else nu[j]))))
matplot(rho, lam.rho, type = "l", lty = 1, lwd = 2, col = 1:n.nu,
        xlab = quote(rho), ylab = quote(lambda))
legend("topleft", legend = expr.rho, bty = "n", lwd = 2, col = 1:n.nu)
## Coefficient of tail dependence as a function of nu
nu. <- c(seq(3, 12, by = 0.2), Inf)
rho. <- c(-1, -0.5, 0, 0.5, 1)
n.rho <- length(rho.)
lam.nu <- sapply(rho., function(rh) # (nu, rho) matrix
  sapply(nu., function(nu) lambda(tCopula(rh, df = nu))[["lower"]]))
expr <- as.expression(lapply(1:n.rho, function(j) bquote(rho == .(rho.[j]))))
matplot(nu., lam.nu, type = "l", lty = 1, lwd = 2, col = 1:n.rho,
        xlab = quote(nu), ylab = quote(lambda))
legend("right", expr, bty = "n", lwd = 2, col = 1:n.rho)


### Effect of rho and nu on P(U_1 > u, U_2 > u) for t copulas

## Note: All calculations here are deterministic
u <- seq(0.95, to = 0.9999, length.out = 128) # levels u of P(U_1> u, U_2> u)
rho <- c(0.75, 0.5) # correlation parameter rho
nu <- c(3, 4, 8, Inf) # degrees of freedom
len <- length(rho) * length(nu)
tail.prob <- matrix(u, nrow = length(u), ncol = 1 + len) # tail probabilities
expr <- vector("expression", length = len) # vector of expressions
ltys <- cols <- numeric(len) # line types and colors
for(i in seq_along(rho)) { # rho
  for(j in seq_along(nu)) { # degrees of freedom
    k <- length(nu) * (i - 1) + j
    ## Create the copula
    cop <- ellipCopula("t", param = rho[i], df = nu[j])
    ## Evaluate P(U_1 > u, U_2 > u) = P(U_1 <= 1 - u, U_2 <= 1 - u)
    tail.prob[,k+1] <- pCopula(cbind(1 - u, 1 - u), copula = cop)
    ## Create plot information
    expr[k] <- as.expression(
      substitute(group("(",list(rho, nu), ")") ==
                   group("(", list(RR, NN), ")"),
                 list(RR = rho[i],
                      NN = if(is.infinite(nu[j]))
                        quote(infinity) else nu[j])))
    ltys[k] <- length(rho) - i + 1
    cols[k] <- j
  }
}
## Standardize w.r.t. Gauss case
tail.prob.fact <- tail.prob # for comparison to Gauss case
tail.prob.fact[,2:5] <- tail.prob[,2:5] / tail.prob[,5]
tail.prob.fact[,6:9] <- tail.prob[,6:9] / tail.prob[,9]
## Plot tail probabilities
matplot(tail.prob[,1], tail.prob[,-1], type = "l", lwd = 2, lty = ltys,
        col = cols, xlab = quote(P(U[1]>u, U[2]>u)~~"as a function of u"),
        ylab = "")
legend("topright", expr, bty = "n", lwd = 2, lty = ltys, col = cols)
## Plot standardized tail probabilities
matplot(tail.prob.fact[,1], tail.prob.fact[,-1], log = "y", type = "l",
        lty = ltys, col = cols, lwd = (wd <- 2*c(1,1,1,1.6,1,1,1,1)),
        xlab = quote(P(U[1]>u, U[2]>u)~~
                       "as a function of u standardized by Gauss case"),
        ylab = "")
legend("topleft", expr, bty = "n", lwd = wd, lty = ltys, col = cols)


### Effect of rho and nu on P(U_1 > 0.99, .., U_d > 0.99) for t copulas

d <- 2:20 # dimensions
u <- 0.99 # level u of P(U_1 > u, ..., U_d > u)
tail.pr.d <- matrix(d, nrow = length(d), ncol = 1+len)# tail prob; P[,1] = d
set.seed(271) # set seed due to MC randomness here
for(i in seq_along(rho)) { # rho
  for(j in seq_along(nu)) { # degrees of freedom
    k <- length(nu) * (i-1) + j
    for(l in seq_along(d)) { # dimension
      ## Create the copula
      cop <- ellipCopula("t", param = rho[i], dim = d[l], df = nu[j])
      ## Evaluate P(U_1 > u,...,U_d > u) = P(U_1 <= 1-u,...,U_d <= 1-u)
      tail.pr.d[l, k+1] <- pCopula(rep(1-u, d[l]), copula = cop)
    }
  }
}
## Standardize w.r.t. Gauss case
tail.pr.d.fact <- tail.pr.d # for comparison to Gauss case
tail.pr.d.fact[,2:5] <- tail.pr.d[,2:5] / tail.pr.d[,5]
tail.pr.d.fact[,6:9] <- tail.pr.d[,6:9] / tail.pr.d[,9]
## Plot tail probabilities
matplot(tail.pr.d[,1], tail.pr.d[,-1], type = "l", log = "y", yaxt = "n",
        lty = ltys, col = cols, lwd = 2, ylab = "",
        xlab = quote(P(U[1] > 0.99, ..., U[d] > 0.99)~~
                       "as a function of d"))
sfsmisc::eaxis(2, cex.axis = 0.8)
axis(1, at = 2)
legend("topright",   expr[1:4], bty="n", lty=ltys[1:4], col=cols[1:4], lwd=2)
legend("bottomleft", expr[5:8], bty="n", lty=ltys[5:8], col=cols[5:8], lwd=2)
## Plot standardized tail probabilities
matplot(tail.pr.d.fact[,1], tail.pr.d.fact[,-1], log = "y", type = "l",
        las = 1, lty = ltys, col = cols,
        lwd = (wd <- 2*c(1,1,1,1.6,1,1,1,1)), ylab = "",
        xlab = quote(P(U[1] > 0.99,..,U[d] > 0.99)~~
                       "as a function of d standardized by Gauss case"))
legend("topleft", expr, bty = "n", lty = ltys, lwd = wd, col = cols)
axis(1, at = 2)

## Joint exceedance probability under the normal copula
d <- 5
rho <- 0.5
u <- 0.99
set.seed(271)
ex.prob.norm <- pCopula(rep(1 - u, d), copula = normalCopula(rho, dim = d))
1 / (260 * ex.prob.norm) # ~ 51.72 years

## Joint exceedance probability under the t copula model with 3 df
## 1) Via scaling of the probability obtained from the normal copula
##    Note that the scaling factor was read off from the previous plot
1 / (2600 * ex.prob.norm) # ~ 5.17 years

## 2) Directly using the t copula
ex.prob.t3 <- pCopula(rep(1 - u, d), copula = tCopula(rho, dim = d, df = 3))
1 / (260 * ex.prob.t3) # ~ 5.91 years


### 2.7 Rosenblatt transform and conditional sampling ##########################

### Evaluation of and sampling from C_{j|1,..,j-1}(.|u_1,..,u_{j-1})

## Define the copula
nu <- 3.5
theta <- iTau(tCopula(df = nu), tau = 0.5)
tc <- tCopula(theta, df = nu)
## Evaluate the df C(.|u_1) at u for several u_1
u <- c(0.05, 0.3, 0.7, 0.95)
u2 <- seq(0, 1, by = 0.01)
ccop <- sapply(u, function(u.)
  cCopula(cbind(u., u2), copula = tc, indices = 2))
## Evaluate the function C(u_2|.) at u for several u_2
u1 <- seq(0, 1, by = 0.01)
ccop. <- sapply(u, function(u.)
  cCopula(cbind(u1, u.), copula = tc, indices = 2))

matplot(ccop, type = "l", lty = 1, lwd = 2,
        col = (cols <- seq_len(ncol(ccop))), ylab = "",
        xlab = substitute(C["2|1"](u[2]~"|"~u[1])~~"as a function of"~
                            u[2]~"for a"~{C^italic(t)}[list(rho,nu)]~"copula",
                          list(nu = nu)))
legend("bottomright", bty = "n", lwd = 2, col = cols,
       legend = as.expression(lapply(seq_along(u), function(j)
         substitute(u[1] == u1, list(u1 = u[j])))))

matplot(ccop., type = "l", lty = 1, lwd = 2,
        col = (cols <- seq_len(ncol(ccop.))), ylab = "",
        xlab = substitute(C["2|1"](u[2]~"|"~u[1])~~"as a function of"~
                            u[1]~"for a"~{C^italic(t)}[list(rho,nu)]~"copula",
                          list(nu = nu)))
legend("center", bty = "n", lwd = 2, col = cols,
       legend = as.expression(lapply(seq_along(u), function(j)
         substitute(u[2] == u2, list(u2 = u[j])))))

## Sample from C_{2|1}(.|u_1)
set.seed(271)
u2 <- runif(1000)
## Small u_1
u1 <- 0.05
U2 <- cCopula(cbind(u1, u2), copula = tc, indices = 2, inverse = TRUE)
## Large u_1
u1. <- 0.95
U2. <- cCopula(cbind(u1., u2), copula = tc, indices = 2, inverse = TRUE)
plot(U2, ylab = substitute(U[2]~"|"~U[1]==u, list(u = u1)))
plot(U2., ylab = substitute(U[2]~"|"~U[1]==u, list(u = u1.)))


### Rosenblatt transform

## Sample from a Gumbel-Hougaard copula
gc <- gumbelCopula(2)
set.seed(271)
U <- rCopula(1000, copula = gc)
## Apply the transformation R_C with the correct copula
U. <- cCopula(U, copula = gc)
## Apply the transformation R_C with a wrong copula
gc. <- setTheta(gc, value = 4) # larger theta
U.. <- cCopula(U, copula = gc.)
plot(U.,  xlab = quote(U*"'"[1]), ylab = quote(U*"'"[2]))
plot(U.., xlab = quote(U*"'"[1]), ylab = quote(U*"'"[2]))


### Conditional distribution method and quasi-random copula sampling

## Define the Clayton copula to be sampled
cc <- claytonCopula(2)
## Pseudo-random sample from the Clayton copula via CDM
set.seed(271)
U.pseudo <- rCopula(1000, copula = indepCopula())
U.cc.pseudo <- cCopula(U.pseudo, copula = cc, inverse = TRUE)
## Quasi-random sample from the Clayton copula via CDM
set.seed(271)
library(qrng)
U.quasi <- ghalton(1000, d = 2) # sobol() is typically even faster
U.cc.quasi <- cCopula(U.quasi, copula = cc, inverse = TRUE)
plot(U.pseudo,    xlab = quote(U*"'"[1]), ylab = quote(U*"'"[2]))
plot(U.quasi,     xlab = quote(U*"'"[1]), ylab = quote(U*"'"[2]))
plot(U.cc.pseudo, xlab = quote(U[1]),     ylab = quote(U[2]))
plot(U.cc.quasi,  xlab = quote(U[1]),     ylab = quote(U[2]))


### Variance reduction

##' @title Approximately computing P(U_1 > u_1,..., U_d > u_d)
##' @param n sample size
##' @param copula copula of (U_1,..., U_d)
##' @param u lower-left endpoint (u_1,..., u_d) of the evaluation point
##' @return Estimates of P(U_1 > u_1,..., U_d > u_d) by
##'         pseudo-random numbers, Latin hypercube sampling and
##'         randomized quasi-random numbers.
sProb <- function(n, copula, u) # sample size, copula, lower-left endpoint
{
  d <- length(u)
  stopifnot(n >= 1, inherits(copula, "Copula"), 0 < u, u < 1,
            d == dim(copula))
  umat <- rep(u, each = n)
  ## Pseudo-random numbers
  U <- rCopula(n, copula = copula)
  PRNG <- mean(rowSums(U > umat) == d)
  ## Latin hypercube sampling (based on the recycled 'U')
  U. <- rLatinHypercube(U)
  LHS <- mean(rowSums(U. > umat) == d)
  ## (Randomized) quasi-random numbers
  U.. <- cCopula(sobol(n, d = d, randomize = TRUE), copula = copula,
                 inverse = TRUE)
  QRNG <- mean(rowSums(U.. > umat) == d)
  ## Return
  c(PRNG = PRNG, LHS = LHS, QRNG = QRNG)
}
## Simulate the probabilities of falling in (u_1, 1] x ... x (u_d, 1]
library(qrng)
N <- 500 # number of replications
n <- 5000 # sample size
d <- 5 # dimension
nu <- 3 # degrees of freedom
rho <- iTau(tCopula(df = nu), tau = 0.5) # correlation parameter
cop <- tCopula(param = rho, dim = d, df = nu) # t copula
u <- rep(0.99, d) # lower-left endpoint of the considered cube
set.seed(271) # for reproducibility
res <- replicate(N, sProb(n, copula = cop, u = u))
## Grab out the results and compute the sample variances
varP <- var(PRNG <- res["PRNG",])
varL <- var(LHS  <- res["LHS" ,])
varQ <- var(QRNG <- res["QRNG",])
## Compute the VRFs and % improvements w.r.t. PRNG
VRF.L <- varP / varL # VRF for LHS
VRF.Q <- varP / varQ # VRF for QRNG
PIM.L <- (varP - varL) / varP * 100 # % improvement for LHS
PIM.Q <- (varP - varQ) / varP * 100 # % improvement for QRNG
## Box plot
boxplot(list(PRNG = PRNG, LHS = LHS, QRNG = QRNG),
        sub = sprintf("N = %d replications with n = %d and d = %d", N, n, d))
mtext(substitute("Simulated"~~P(bold(U) > bold(u))~~
                   "for a"~{C^italic(t)}[list(rho.,nu.)]~"copula",
                 list(rho. = round(rho, 2), nu. = nu)),
      side = 2, line = 4.5, las = 0)
mtext(sprintf("VRFs (%% improvements): %.1f (%.0f%%), %.1f (%.0f%%)",
              VRF.L, PIM.L, VRF.Q, PIM.Q),
      side = 4, line = 1, adj = 0, las = 0)








#http://firsttimeprogrammer.blogspot.com/2016/03/how-to-fit-copula-model-in-r-heavily.html

#The most common Archimedean copulas families are Frank, Gumbel and Clayton. 




# Copula package
library(copula)
# Fancy 3D plain scatterplots
library(scatterplot3d)
# ggplot2
library(ggplot2)
# Useful package to set ggplot plots one next to the other
library(grid)
set.seed(235)




# Generate a bivariate normal copula with rho = 0.7
normal <- normalCopula(param = 0.7, dim = 2)
normal
# Generate a bivariate t-copula with rho = 0.8 and df = 2
stc <- tCopula(param = 0.8, dim = 2, df = 2)

stc

# Build a Frank, a Gumbel and a Clayton copula
frank <- frankCopula(dim = 2, param = 8)
gumbel <- gumbelCopula(dim = 3, param = 5.6)
clayton <- claytonCopula(dim = 4, param = 19)

# Print information on the Frank copula
print(frank)

# Select the copula
cp <- claytonCopula(param = c(3.4), dim = 2)

# Generate the multivariate distribution (in this case it is just bivariate) with normal and t marginals
multivariate_dist <- mvdc(copula = cp,
                          margins = c("norm", "t"),
                          paramMargins = list(list(mean = 2, sd=3),
                                              list(df = 2)) )
print(multivariate_dist)

# Generate random samples
fr <- rCopula(2000, frank)
gu <- rCopula(2000, gumbel)
cl <- rCopula(2000, clayton)

# Plot the samples
p1 <- qplot(fr[,1], fr[,2], colour = fr[,1], main="Frank copula random samples theta = 8", xlab = "u", ylab = "v")
p2 <- qplot(gu[,1], gu[,2], colour = gu[,1], main="Gumbel copula random samples theta = 5.6", xlab = "u", ylab = "v") 
p3 <- qplot(cl[,1], cl[,2], colour = cl[,1], main="Clayton copula random samples theta = 19", xlab = "u", ylab = "v")


frank

# Define grid layout to locate plots and print each graph^(1)
pushViewport(viewport(layout = grid.layout(1, 3)))
print(p1, vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
print(p2, vp = viewport(layout.pos.row = 1, layout.pos.col = 2))
print(p3, vp = viewport(layout.pos.row = 1, layout.pos.col = 3))

samples <- rMvdc(2000, multivariate_dist)
scatterplot3d(samples[,1], samples[,2], color = "blue",pch = ".")












# Generate the normal copula and sample some observations
coef_ <- 0.5
mycopula <- normalCopula(coef_, dim = 2)
u <- rCopula(2000, mycopula)

cor(u[,1], u[,2])

# Compute the density
pdf_ <- dCopula(u, mycopula)

# Compute the CDF
cdf <- pCopula(u, mycopula)

# Generate random sample observations from the multivariate distribution
v <- rMvdc(2000, multivariate_dist)

# Compute the density
pdf_mvd <- dMvdc(v, multivariate_dist)

# Compute the CDF
cdf_mvd <- pMvdc(v, multivariate_dist)









par(mfrow = c(1, 3))
# 3D plain scatterplot of the density, plot of the density and contour plot
scatterplot3d(u[,1], u[,2], pdf_, color="red", main="Density", xlab ="u1", ylab="u2", zlab="dCopula", pch=".")
persp(mycopula, dCopula, main ="Density")
contour(mycopula, dCopula, xlim = c(0, 1), ylim=c(0, 1), main = "Contour plot")

par(mfrow = c(1, 3))
# 3D plain scatterplot of the CDF, plot of the CDF and contour plot
scatterplot3d(u[,1], u[,2], cdf, color="red", main="CDF", xlab = "u1", ylab="u2", zlab="pCopula",pch=".")
persp(mycopula, pCopula, main = "CDF")
contour(mycopula, pCopula, xlim = c(0, 1), ylim=c(0, 1), main = "Contour plot")

# 3D plain scatterplot of the multivariate distribution
par(mfrow = c(1, 2))
scatterplot3d(v[,1],v[,2], pdf_mvd, color="red", main="Density", xlab = "u1", ylab="u2", zlab="pMvdc",pch=".")
scatterplot3d(v[,1],v[,2], cdf_mvd, color="red", main="CDF", xlab = "u1", ylab="u2", zlab="pMvdc",pch=".")
persp(multivariate_dist, dMvdc, xlim = c(-4, 4), ylim=c(0, 2), main = "Density")
contour(multivariate_dist, dMvdc, xlim = c(-4, 4), ylim=c(0, 2), main = "Contour plot")
persp(multivariate_dist, pMvdc, xlim = c(-4, 4), ylim=c(0, 2), main = "CDF")
contour(multivariate_dist, pMvdc, xlim = c(-4, 4), ylim=c(0, 2), main = "Contour plot")




frank <- frankCopula(dim = 2, param = 3)
clayton <- claytonCopula(dim = 2, param = 1.2)
gumbel <- gumbelCopula(dim = 2, param = 1.5)

par(mfrow = c(1, 3))

# Density plot
persp(frank, dCopula, main ="Frank copula density")
persp(clayton, dCopula, main ="Clayton copula density")
persp(gumbel, dCopula, main ="Gumbel copula density")

# Contour plot of the densities
contour(frank, dCopula, xlim = c(0, 1), ylim=c(0, 1), main = "Contour plot Frank")
contour(clayton, dCopula, xlim = c(0, 1), ylim=c(0, 1), main = "Contour plot Clayton")
contour(gumbel, dCopula, xlim = c(0, 1), ylim=c(0, 1), main = "Contour plot Gumbel")









library(copula)
library(psych)
# set.seed(100)
myCop <- normalCopula(param=c(0.8), dim = 2, dispstr = "un")
myMvd <- mvdc(copula=myCop, margins=c("exp", "exp"),
              paramMargins=list(list(rate=1),
                                list(rate=1)) )

Z2 <- rMvdc(2000,myMvd)
Z2
colnames(Z2) <- c("x1", "x2")
pobs(Z2)
library(VineCopula)
u <- pobs(Z2)[,1]
v <- pobs(Z2)[,2]

selectedCopula <- BiCopSelect(u,v,familyset=NA)
selectedCopula

n.cop <- normalCopula(dim=2)
# set.seed(500)
# m <- Z2
fit <- fitCopula(n.cop,pobs(Z2),method='ml')





