#-----------------------------------------------------------------------------------------
# MAKING THE FULL DATASET
dat <- cbind(y,x) # THIS IS FD
#-------------------------------------------------------------------------------------
# MAKING THE REDUCED DATASET USING LASSO
xrq=model.matrix(y~.,dat)[,-1] # take out the first column which are all 1's for intercept
yrq = dat$y
xrq=scale(xrq)
lasso.dat <- cbind(y,xrq)
library(glmnet)
set.seed(123)
cv.lasso=cv.glmnet(xrq,yrq,family="binomial",type.measure = "class",alpha=1)
plot(cv.lasso)
lasso.best.lambda=cv.lasso$lambda.min # find the best lambda value corresponding to min cv.error
lasso.lambda.1se=cv.lasso$lambda.1se 
lasso.coef=predict (cv.lasso ,type="coefficients",s= lasso.best.lambda) [1:501,]
lasso.coef.nonzero = lasso.coef[lasso.coef!=0]
# USING all the columns whose coefficients do not go to zero above
crq <- c("y","V4","V45","V111","V179","V191","V202","V203","V224","V249","V276","V301","V401","V405","V442","V480","V483")
lasso.best.dat <- dat[crq] #THIS IS RD
############################################################################
########################################################################################################
# Datasets for Cross Validation
set.seed(123)
n = nrow(dat) 
K = 10
folds = sample(1:K, n, replace=TRUE)
##########################################################################################
#RANDOM FOREST ON RD
library(randomForest)
require(rfUtilities)
set.seed(123)
rf.tune = tune.randomForest(x = lasso.best.dat[,-1],y = as.factor(lasso.best.dat[,1]),ntree = c(10000),mtry = c(1:13))
set.seed(123)
rf.test.error = rep(0,K)
rf.oob.error = rep(0,K)
CV.error.rf = rep(0,K)
rf.oob.error.mean = rep(0,K)
for (m in 1:K){
  train.data <- lasso.best.dat[folds!=m,]
  test.data <-  lasso.best.dat[folds==m,]
  rf.dat=randomForest(as.factor(train.data$y)~.,data=train.data,ntrees=rf.tune$best.parameters$ntree,mtry=rf.tune$best.parameters$mtry)
  rf.pred = predict(rf.dat,test.data)
  rf.oob.error[m] = rf.dat$err.rate[500,1]
  rf.test.error[m] = mean(rf.pred != test.data$y) 
}
rf.oob.error.mean = mean(rf.oob.error)
CV.error.rf = mean(rf.test.error)
##############################################################################################
###############################################################################################
# BEST MODEL ON THE TESTING DATA
set.seed(123)
crq_test <- c("V4","V45","V111","V179","V191","V202","V203","V224","V249","V276","V301","V401","V405","V442","V480","V483")
testing <- xnew[crq_test]
rf.train.dat=randomForest(as.factor(lasso.best.dat$y)~.,data=lasso.best.dat,ntrees=10000,mtry=rf.tune$best.parameters$mtry)
ynew = predict(rf.train.dat,testing)
save(ynew,CV.error.rf,file="F:/TAMU/Data Mining&Analysis/Project/830000167.RData")
