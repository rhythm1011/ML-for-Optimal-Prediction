#SUPERVISED LEARNING TASK####################################################################################
#----------------------------------------------------------------------------------------------------------
# THERE ARE TWO DATASETS IN THE FOLLOWING ANALYSIS. RD WHICH IS LABELLED AS "lasso.best.dat" WHICH HAS ONLY 21 FEATURES
# AND "dat" WHICH HAS ALL FEATURES
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
lasso.coef.nonzero
crq <- c("y","V4","V45","V111","V179","V191","V202","V203","V224","V249","V276","V301","V401","V405","V442","V480","V483")
lasso.best.dat <- dat[crq] #THIS IS RD
########################################################################################################
# Datasets for Cross Validation
set.seed(123)
n = nrow(dat) 
K = 10
folds = sample(1:K, n, replace=TRUE)
########################################################################################################
# LINEAR DISCRIMINANT ANALYSIS
# USE "lasso.best.dat" for RD and "dat" for FD
set.seed(123)
require(MASS)
lda.test.error = rep(0,K)
for (m in 1:K){
  train.data <- lasso.best.dat[folds!=m,]
  test.data <- lasso.best.dat[folds==m,]
  lda.fit=lda(train.data$y~.,data=train.data)
  lda.pred=predict(lda.fit,test.data)
  lda.test.error[m] = mean(lda.pred$class!=test.data$y)
}
CV.error.lda = mean(lda.test.error)

####################################################################################################
# TREES: Tuning for pruning depth
# USE "lasso.best.dat" for RD and "dat" for FD
library(tree)
set.seed(123)
test.error.tree = rep(0,K)
CV.error.tree = rep(0,20)
for (b in 2:20){
  for (m in 1:K){
    test.data <- lasso.best.dat[folds==m,]
    train.data <- lasso.best.dat[folds!=m,]
    tree.a=tree(as.factor(train.data$y)~.,data=train.data)
    tree.cv = cv.tree(tree.a, FUN = prune.misclass)
    prune.tree = prune.misclass(tree.a, best = b)
    tree.pred = predict(prune.tree, test.data, type="class")
    test.error.tree[m] = mean(tree.pred != test.data$y)
  }
  CV.error.tree[b] = mean(test.error.tree)
}
min(CV.error.tree)
############################################################################################
#BOOSTING Tuning for n.trees, interaction.depth, shrinkage
# USE "lasso.best.dat" for RD and "dat" for FD
library(caret)
library(gbm)
set.seed(123)
caretGrid <- expand.grid(interaction.depth=c(1,3,4,2,5), n.trees = (40:100)*10, shrinkage=c(0.01, 0.001),n.minobsinnode=10)
trainControl <- trainControl(method="cv", number=10)
gbm.caret <- caret::train(as.factor(y)~.,data= lasso.best.dat,distribution="bernoulli", method="gbm",trControl=trainControl, verbose=FALSE,tuneGrid=caretGrid, bag.fraction=0.75)

#################################################################################################
#BOOSTING
# USE "lasso.best.dat" for RD and "dat" for FD
library(gbm)
set.seed(123)
test.error = rep(0, K)
for (m in 1:K){
  train.data <- lasso.best.dat[folds!=m,]
  test.data <- lasso.best.dat[folds==m,]
  
  boost.classdat=gbm(train.data$y~.,data=train.data,distribution="bernoulli",n.trees=gbm.caret$bestTune$n.trees,shrinkage=gbm.caret$bestTune$shrinkage,interaction.depth=gbm.caret$bestTune$interaction.depth)
  gbmpred=predict(boost.classdat,newdata=test.data[,2:17],n.trees = gbm.caret$bestTune$n.trees,type="response") 
  pred.class = ifelse(gbmpred>0.5,1,0)
  test.error[m] = mean(pred.class != test.data$y)
}
CV.error.boost = mean(test.error)

##########################################################################################
# SUPPORT VECTOR MACHINE
# USE "lasso.best.dat" for RD and "dat" for FD

library(e1071)
set.seed(123)
svm.tune.poly = tune.svm(x = lasso.best.dat[,-1],y = as.factor(lasso.best.dat[,1]),kernel="polynomial",cost = c(0.001,0.01,0.1,1,10,100),degree = c(1,2,3,4,5)) 
svm.tune.rad = tune.svm(x = lasso.best.dat[,-1],y = as.factor(lasso.best.dat[,1]),kernel="radial",cost = c(0.001,0.01,0.1,1,10,100),gamma=c(0.001,0.01,0.1,1,10,100))
svm.tune.lin = tune.svm(x = lasso.best.dat[,-1],y = as.factor(lasso.best.dat[,1]),kernel="linear",cost = c(0.001,0.01,0.1,1,10,100))
svm.tune.sig = tune.svm(x = lasso.best.dat[,-1],y = as.factor(lasso.best.dat[,1]),kernel="sigmoid",cost = c(0.001,0.01,0.1,1,10,100),gamma=c(0.001,0.01,0.1,1,10,100))
#Change kernel and add the values from tune accordingly
set.seed(123)
svm.test.error = rep(0,K)
for (m in 1:K){
  train.data <- lasso.best.dat[folds!=m,]
  test.data <- lasso.best.dat[folds==m,]
  svmfit=svm(train.data$y~.,data=train.data, kernel="radial",gamma = 0.001,cost=1,type="C-classification")
  svm.pred=predict(svmfit, test.data)
  svm.test.error[m] = sum(svm.pred!=test.data$y)/length(test.data$y)
}
CV.error.svm = mean(svm.test.error)

########################################################################################################
# RANDOM FOREST
# USE "lasso.best.dat" for RD and "dat" for FD

library(randomForest)
require(rfUtilities)
set.seed(123)
rf.tune = tune.randomForest(x = lasso.best.dat[,-1],y = as.factor(lasso.best.dat[,1]),ntree = c(1000),mtry = c(1:15))

rf.test.error = rep(0,K)
rf.oob.error = rep(0,K)
CV.error.rf = rep(0,K)
rf.oob.error.mean = rep(0,K)
for (m in 1:K){
  train.data <- lasso.best.dat[folds!=m,]
  test.data <-  lasso.best.dat[folds==m,]
  rf.dat=randomForest(as.factor(train.data$y)~.,data=train.data,ntrees=1000,mtry=rf.tune$best.parameters$mtry)
  rf.pred = predict(rf.dat,test.data)
  rf.oob.error[m] = rf.dat$err.rate[500,1]
  rf.test.error[m] = mean(rf.pred != test.data$y) 
}
rf.oob.error.mean = mean(rf.oob.error)
CV.error.rf = mean(rf.test.error)


#######################################################################################################
#LOGISTIC REGRESSION
# USE "lasso.best.dat" for RD and "dat" for FD
set.seed(123)
lr.test.error=rep(0,K)
for(m in 1:K){
  test.data <- lasso.best.dat[folds==m,]
  train.data <- lasso.best.dat[folds!=m,]
  mylogit <- glm(y~., data = train.data, family = "binomial",maxit=1000000000)
  glm.probs=predict(mylogit,test.data,type="response")
  glm.pred=ifelse(glm.probs>0.5,"1","0")
  lr.test.error[m] = mean(glm.pred!=test.data$y)
}
CV.error.lr = mean(lr.test.error)
#####################################################################################################
# TRIED NEURAL NETWORKS
require(neuralnet)
set.seed(123)
nn.test.error=rep(0,K)
for(m in 1:K){
  test.data <- lasso.best.dat[folds==m,]
  train.data <- lasso.best.dat[folds!=m,]
  nn=neuralnet(as.factor(y)~., data=train.data, hidden=3,act.fct = "logistic",linear.output = FALSE)
  pred = compute(nn,test.data)
  nn.prob <- pred$net.result
  nn.pred=ifelse(nn.prob>0.5,"1","0")
  nn.test.error[m] = mean(nn.pred!=test.data$y)
}
CV.error.nn = mean(nn.test.error)



#######################################################################################################
#####################################################################################################
#UNSUPERVISED LEARNING TASK##############################################################
#----------------------------------------------------------------------------------------------------
#Load packages, then get started
library(fpc)
library(dbscan)
if(!require(devtools)) install.packages('devtools')
devtools::install_github('kassambara/factoextra')
library(factoextra)
library(mclust)
?library(cluster)
library(NbClust)
library(gridExtra)
library(ggplot2)
library(clValid)

#Load the data-----------------------------------------------------------------------
#PCA
clusterdata = load("G:/My Drive/Class/STAT 639/Project/cluster_data(1).RData")

#set the seed
set.seed(123)

#------------
pca.out=prcomp(y, scale=TRUE) 
pca.out 
apply(pca.out$x, 2, sd)
pr.var=pca.out$sdev^2
pve <-pr.var/sum(pr.var)
plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')
plot(cumsum(pve), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1),xlim = c(0,750),type='b')
cs <- cumsum(pve)

#create y90 and y95 datasets which account for 90 & 95% of pve 
y90 <- pca.out$x[,1:min(which(cs>0.90))]
d90 <- dist(y90)
y95 <- pca.out$x[,1:min(which(cs>0.95))]
d95 <- dist(y95)

#-----------------------------------------Determine Optimal K------------------------------------------
#Use NbClust and gap statistic as measure of optimal K

testindex <- 'all'
test90.comp <- NbClust(y90,method = 'complete',index = testindex,min.nc = 2,max.nc = 50) 
test95.comp <- NbClust(y95,method = 'complete',index = testindex,min.nc = 2,max.nc = 50)
gs90.comp <- clusGap(y90, FUN = hcut, nstart = 25, K.max = 50, B = 50)
gs95.comp <- clusGap(y95, FUN = hcut, nstart = 25, K.max = 50, B = 50)

test90.km <- NbClust(y90,method = 'kmeans',index = testindex,min.nc = 2,max.nc = 50) 
test95.km <- NbClust(y95,method = 'kmeans',index = testindex,min.nc = 2,max.nc = 50)
gs90.km <- clusGap(y90, FUN = kmeans, nstart = 25, K.max = 50, B = 50)
gs95.km <- clusGap(y95, FUN = kmeans, nstart = 25, K.max = 50, B = 50)

#----------Determining optimal k Results----------
#-----hierarchical-----
#nbclust (hierarchical)
c1 <- factoextra::fviz_nbclust(test90.comp)+ ggtitle("Hierarchical")
c2 <- factoextra::fviz_nbclust(test95.comp)
grid.arrange(c1, c2,nrow = 2)

#WSS, Silhouette, Gap Statistic (hierarchical)
hy90.wss <- fviz_nbclust(y90, hcut,method = 'wss', k.max = 50) +ggtitle('Y90')
hy90.sil <- fviz_nbclust(y90, hcut,method = 'silhouette', k.max = 50)
hy90.gps <- fviz_gap_stat(gs90.comp)
grid.arrange(hy90.wss,hy90.sil,hy90.gps,nrow = 3)

hy95.wss <- fviz_nbclust(y95, hcut,method = 'wss', k.max = 50) +ggtitle('Y95')
hy95.sil <- fviz_nbclust(y95, hcut,method = 'silhouette', k.max = 50)
hy95.gps <- fviz_gap_stat(gs95.comp)
grid.arrange(hy95.wss,hy95.sil,hy95.gps,nrow = 3)

#-----k-means-----
#nbclust (kmeans)
k1 <- factoextra::fviz_nbclust(test90.km)+ ggtitle("NbClust Results: Y90") +
  theme(text = element_text(size = 25),
        axis.title = element_text(size = 25),
        axis.text = element_text(size = 18)) 
k2 <- factoextra::fviz_nbclust(test95.km)+ ggtitle("NbClust Results: Y95") +
  theme(text = element_text(size = 25),
        axis.title = element_text(size = 25),
        axis.text = element_text(size = 18)) 
grid.arrange(k1, k2,nrow = 2)

#WSS, Silhouette, Gap Statistic (kmeans)
y90.wss <- fviz_nbclust(y90, kmeans,method = 'wss', k.max = 50) +ggtitle('Y90')
y90.sil <- fviz_nbclust(y90, kmeans,method = 'silhouette', k.max = 50) + ggtitle('Optimal Number of Clusters Y90') +
  theme(text = element_text(size = 25),
        axis.title = element_text(size = 25),
        axis.text = element_text(size = 18))
y90.gps <- fviz_gap_stat(gs90.km)

y95.wss <- fviz_nbclust(y95, kmeans,method = 'wss', k.max = 50) +ggtitle('Y95')
y95.sil <- fviz_nbclust(y95, kmeans,method = 'silhouette', k.max = 50)+ ggtitle('Optimal Number of Clusters Y95') +
  theme(text = element_text(size = 25),
        axis.title = element_text(size = 25),
        axis.text = element_text(size = 18))
y95.gps <- fviz_gap_stat(gs95.km)

grid.arrange(y90.wss,y90.sil,y90.gps,nrow = 3)
grid.arrange(y95.wss,y95.sil,y95.gps,nrow = 3)
grid.arrange(y90.sil,y95.sil,nrow = 2)

#AIC & BIC (kmeans)
kmeansAIC = function(fit){
  m = ncol(fit$centers)
  n = length(fit$cluster)
  k = nrow(fit$centers)
  D = fit$tot.withinss
  return(data.frame(AIC = D + 2*m*k,
                    BIC = D + log(n)*m*k))
}
kmeansAIC_vec90 <- rep(0, 50)
kmeansBIC_vec90 <- rep(0, 50)
kmeansAIC_vec95 <- rep(0, 50)
kmeansBIC_vec95 <- rep(0, 50)

for (i in 2:50) {
  km.fit90 <- kmeansAIC(kmeans(y90,centers = i, nstart = 50))
  kmeansAIC_vec90[i] <- km.fit90$AIC
  kmeansBIC_vec90[i] <- km.fit90$BIC
  km.fit95 <- kmeansAIC(kmeans(y95,centers = i, nstart = 50))
  kmeansAIC_vec95[i] <- km.fit95$AIC
  kmeansBIC_vec95[i] <- km.fit95$BIC}

{plot(2:50, kmeansAIC_vec90[2:50],type = 'b', xlab = 'Numeber of Clusters', ylab = 'AIC',main = 'y90')
  abline(v = which.min(kmeansAIC_vec90[2:50])+1)
  plot(2:50, kmeansBIC_vec90[2:50],type = 'b', xlab = 'Numeber of Clusters', ylab = 'BIC')
  abline(v = which.min(kmeansBIC_vec90[2:50])+1)
  which.min(kmeansBIC_vec90[2:50])+1}

{plot(2:50, kmeansAIC_vec95[2:50],type = 'b', xlab = 'Numeber of Clusters', ylab = 'AIC', main = 'y95')
  abline(v = which.min(kmeansAIC_vec95[2:50])+1)
  plot(2:50, kmeansBIC_vec95[2:50],type = 'b', xlab = 'Numeber of Clusters', ylab = 'BIC')
  abline(v = which.min(kmeansBIC_vec95[2:50])+1)
  which.min(kmeansBIC_vec95[2:50])+1}

#----------------------------------------Clustering Results-------------------------------------------
#k-means clusters
km90 <- kmeans(y90,centers = 2, nstart = 50) 
fviz_cluster(object = list(data = y90,cluster = test90.km$Best.partition),geom='point') + ggtitle('K means Cluster Plot: Y90') + 
  theme(text = element_text(size = 25),
        axis.title = element_text(size = 25),
        axis.text = element_text(size = 18))

km95 <- kmeans(y95,centers = 2, nstart = 50)
fviz_cluster(object = list(data = y95,cluster = test95.km$Best.partition),geom='point')  +ggtitle('K means Cluster Plot: Y95') +
  theme(text = element_text(size = 25),
        axis.title = element_text(size = 25),
        axis.text = element_text(size = 18))

#hierarchical clustering
hc90<-hclust(d90,method = 'complete')
mem90 <- cutree(hc90,2)

hc95<-hclust(d95)
mem95 <- cutree(hc95,2)

fviz_cluster(mem90)

#density based clustering using DBSCAN
kNNdistplot(y90,k=4,all = TRUE)
abline(h =30,lty = 2)

kNNdistplot(y95,k=4,all = TRUE)
abline(h =33,lty = 2)

f90 <-fpc::dbscan(y90,eps = 30,MinPts = 4)
f95 <-fpc::dbscan(y95,eps = 33,MinPts = 4)

fviz_cluster(f90,y90,geom = points)
fviz_cluster(f95,y95,geom = points)

#Gaussian Mixture Models
gmm90 <- Mclust(y90, G = 2:50)
gmm95 <- Mclust(y95, G = 2:50)

gmp90 <- fviz_mclust_bic(gmm90)
gmp95 <- fviz_mclust_bic(gmm95)

fviz_cluster(gmm90,geom = 'point')
fviz_cluster(gmm95,geom = 'point')