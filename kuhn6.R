#kuhn6



###############
library(AppliedPredictiveModeling)
library(caret)
library(MASS)
library(ggplot2)
library(pls)
library(elasticnet)
data(solubility)
names(trainingData)
ls(pattern="solT")

set.seed(2)
sample(names(solTrainX),8)

trainingData<-solTrainXtrans
trainingData$Solubility<-solTrainY
lmFitAllPredictors<-lm(Solubility~.,data=trainingData)
summary(lmFitAllPredictors)

lmPred1<-predict(lmFitAllPredictors,solTestXtrans)
head(lmPred1)

lmValues1<-data.frame(obs=solTestY,pred=lmPred1)
defaultSummary(lmValues1)

rlmFitAllPredictors<-rlm(Solubility~.,data=trainingData)
ctrl<-trainControl(method="cv",number=10)

set.seed(100)
lmFit1<-train(x=solTrainXtrans,y=solTrainY,
             method="lm",trControl=ctrl)

xyplot(solTrainY~predict(lmFit1),
       type=c("p","g"),
       xlab="Predicted",ylab="Observed")

xyplot(resid(lmFit1)~predict(lmFit1),
       type=c("p","g"),
       xlab="Predicted",ylab="Residuals")

corThresh<-.9
tooHigh<-findCorrelation(cor(solTrainXtrans),corThresh)
corrPred<-names(solTrainXtrans)[tooHigh]
trainXfiltered<-solTrainXtrans[,-tooHigh]

testXfiltered<-solTestXtrans[,-tooHigh]
set.seed(100)
lmFiltered<-train(trainXfiltered,solTrainY,method="lm",
                  trControl=ctrl)
lmFiltered

set.seed(100)
rlmPCA<-train(solTrainXtrans,solTrainY,
              method="rlm",
              preProcess="pca",
              trControl=ctrl)
rlmPCA

plsFit<-plsr(Solubility~.,data=trainingData)
predict(plsFit,solTestXtrans[1:5],ncomp=1:2)


set.seed(100)
plsTune<-train(solTrainXtrans,solTrainY,
               method="pls",
               ##the default tuning grid evaluates
               ##components 1...tunelength
               tuneLength=20,
               trControl=ctrl,
               ##put the predictors on the same scale
               preProc=c("center","scale"))


ridgeModel <- enet(x=as.matrix(solTrainXtrans),y=solTrainY,
                   lambda=0.001)
ridgePred<-predict(ridgeModel,newx=as.matrix(solTestXtrans),
                   s=1,mode="fraction",
                   type="fit")
head(ridgePred$fit)

ridgeGrid<-data.frame(.lambda = seq(0,.1,length=15))
set.seed(100)
ridgeRegFit<-train(solTrainXtrans,solTrainY,
                   method="ridge",
                   ##fir the model over many penalty values
                   tuneGrid=ridgeGrid,
                   trControl=ctrl,
                   ##put the predictors on the same scale
                   preProc=c("center","scale"))


enetModel<-enet(x=as.matrix(solTrainXtrans),y=solTrainY,
                lambda=0.01,normalize=TRUE)

enetPred<-predict(enetModel,newx=as.matrix(solTestXtrans),
                  s=.1,mode="fraction",
                  type="fit")
names(enetPred)

head(enetPred$fit)

enetCoef<-predict(enetModel,newx=as.matrix(solTestXtrans),
                  s=.1,mode="fraction",
                  type="coefficients")
tail(enetCoef$coefficients)

enetGrid<-expand.grid(.lambda=c(0,0.01,.1),
                      .fraction=seq(.05,1,length=20))
set.seed(100)
enetTune<-train(solTrainXtrans,solTrainY,
                method="enet",
                tuneGrid=enetGrid,
                trControl=ctrl,
                preProc=c("center","scale"))


#####CH6自己写
#计算表中数值列Spearman Kendall Pearson
x <- trainingData[,c("FP001","FP002","FP003","FP004","FP005","MolWeight","NumAtoms","NumNonHAtoms")]

y <- trainingData[,c("Ship","CSa","SchedOntime","ReUse","DefectsInspect","DefectsTest","FieldedVolatility","Maint","Experience","DomainExp",
                   "Language","DevPlatform","Turnover","ReqVolatility","Parts ","FormalUnitTest","IntegrationTest","SystemTest","UserTrials","BudgetLoss","ScheduleAchievement","DKSLOC",
                   "CustomerBetaTest","TesterType","ChangedPriorityReqs")]

x <- trainingData$CSat
y <- trainingData$Ship
cor(x,y,method="pearson")
cor(x,y,method="spearman")
cor(x,y,method="kendall")


cor(trainingData)

plot(cor(trainingData))


##散点图
plot(trainingData$NumChlorine)
plot(trainingData$MolWeight)
plot(trainingData$NumOxygen)

#拟合一条曲线要知道直线内容，这个直线是拟合曲线？但是要两个变量才能有这种拟合曲线


names(trainingData)

###########CH4
library(AppliedPredictiveModeling)
library(caret)
library(Design)
library(MASS)
library(ipred)
library(nlme)
library(BRugs)
library(lattice)
library(ggplot2)
data(twoClassData)

str(predictors)
str(classes)

set.seed(1)
trainingRows<- createDataPartition(classes, p = .80 , list=FALSE)
head(trainingRows)

trainPredictors<-predictors[trainingRows,]
trainClasses<-classes[trainingRows]

testPredictors<-predictors[-trainingRows,]
testClasses<-classes[-trainingRows]

str(trainPredictors)
str(testPredictors)

set.seed(1)
repeatedSplits<-createDataPartition(trainClasses,p=.80,times=3)
str(repeatedSplits)

set.seed(1)
cvSplits<-createFolds(trainClasses,k=10,
                      returnTrain = TRUE)

str(cvSplits)
fold1<-cvSplits[[1]]

cvPredictors1<-trainPredictors[fold1,]
cvClasses1<-trainClasses[fold1]
nrow(trainPredictors)
nrow(cvPredictors1)

modelFunction(price~numBedrooms+numBaths+acres,data=housingData)
modelFunction(x=housePredictors,y=price)

trainPredictors<-as.matrix(trainPredictors)
knnFit<-knn3(x=trainPredictors,y=trainClasses,k=5)
knnFit

testPredictions<-predict(knnFit,newdata=testPredictors,
                         type="class")
head(testPredictions)
str(testPredictions)

data("GermanCredit")
set.seed(1056)
svmFit<-train(Class~.,data=GermanCreditTrain,
              method="svnRadial")

set.seed(1056)
svmFit<-train(Class~.,data=GermanCreditTrain,
              method="svnRadial",
              preProc=c("center","scale"))

set.seed(1056)
svmFit<-train(Class~.,data=GermanCreditTrain,
              method="svnRadial",
              preProc=c("center","scale"),
              tuneLength=10)


set.seed(1056)
svmFit<-train(Class~.,data=GermanCreditTrain,
              method="svnRadial",
              preProc=c("center","scale"),
              tuneLength=10,
              trControl(method="repeatedcv",
                        repeats=5,
                        classProbs=TRUE))
svmFit

plot(svmFit,scales=list(x=list(log=2)))
predictedClasses<-predict(svmFit,GermanCreditTest)
str(predictedClasses)

predictedProbs<-predict(svmFit,newdata=GermanCreditTest,
                        type="prob")
head(predictedProbs)


set.seed(1056)
logisticReg<-train(Class~.,data=GermanCreditTrain,
                   method="glm",
                   trControl=trainControl(mrthod="repeatedcv",repeats=5))

logisticReg

resamp<-rasamples(list(SVN=svmFit,Logistic=logisticReg))
summary(resamp)

modelDifferences<-diff(resamp)
summary(modelDifferences)


#####CH3
apropos("confusion")
RSiteSearch("confusion",restrict="functions")
library(AppliedPredictiveModeling)
data(segmentationOriginal)
names(segmentationOriginal)

segData<-subset(segmentationOriginal,Case=="Train")
cellID<-segData$Cell
class<-segData$Class
case<-segData$Case
segData<-segData[,-(1:3)]
statusColNum<-grep("Status",names(segData))
statusColNum
segData<-segData[,-statusColNum]

library(e1071)
skewness(segData$AngleCh1)
skewValues<-apply(segData,2,skewness)
head(skewValues)

library(caret)
Ch1AreaTrans<-BoxCoxTrans(segData$AreaCh1)
Ch1AreaTrans

head(segData$AreaCh1)
predict(Ch1AreaTrans,head(segData$AreaCh1))

pcaObject<-prcomp(segData,center=TRUE)
percentVariance<-pcaObject$sd~2/sum(pcaObject$sd~2)*100
percentVariance[1:3]

head(pcaObject$x[,1:5])
head(pcaObject$rotation[,1:3])

trans<-preProcess(segData,method=c("BoxCox","center","scale","pca"))
trans

transformed<-predict(trans,segData)
head(transformed[,1:5])

nearZeroVar(segData)
correlations<-cor(segData)
dim(correlations)

correlations[1:4,1:4]

library(corrplot)
corrplot(correlations,order="hclust")

highCorr<-findCorrelation(correlations,cutoff=.75)
length(highCorr)
head(highCorr)
filteredSegData<-segData[,-highCorr]

library(caret)
data(cars)
type <- c("convertible", "coupe", "hatchback", "sedan", "wagon")
cars$Type <- factor(apply(cars[, 14:18], 1, function(x) type[which(x == 1)]))

carSubset <- cars[sample(1:nrow(cars), 20), c(1, 2, 19)]

head(carSubset)
levels(carSubset$Type)

simpleMod <- dummyVars(~Mileage + Type,
                       data = carSubset,
                       ## Remove the variable name from the
                       ## column name
                       levelsOnly = TRUE)
simpleMod

withInteraction <- dummyVars(~Mileage + Type + Mileage:Type,
                             data = carSubset,
                             levelsOnly = TRUE)
withInteraction
predict(withInteraction, head(carSubset))

library(mlbench)
data(Glass)
str(Glass)




######CH18
library(AppliedPredictiveModeling)
data(solubility)
cor(solTrainXtrans$NumCarbon,solTrainY)
fpCols<-grepl("FP",names(solTrainXtrans))
numericPreds<-names(solTrainXtrans)[!fpCols]
corrValues<-apply(solTrainXtrans[,numericPreds],
                  MARGIN=2,
                  FUN=function(x,y)cor(x,y),
                  y=solTrainY)
head(corrValues)

library(stats)
method="spearman"
smoother<-loess(solTrainY~solTrainXtrans$NumCarcon)
smoother

xyplot(solTrainY~solTrainXtrans$NumCarcon,
       type=c("p","smooth"),
       xlab="# Carbons",
       ylab="Solubility")

loessResults<-filterVarImp(x=solTrainXtrans[,numericPreds],
                           y=solTrainY,
                           nonpara=TRUE)
head(loessResults)

library(minerva)
micValues<-mine(solTrainXtrans[,numericPreds],solTrainY)
names(micValues)
head(micValues$MIC)

t.test(solTrainY~solTrainXtrans$FP044)

getTstats<-function(x,y)
{
  tTest<-t.test(y~x)
  out<-c(tStat=tTest$statistic,p=tTest$p.value)
  out
}

tVals<-apply(solTrainXtrans[,fpCols],
             MARGIN=2,
             FUN=getTstats,
             y=solTrainY)
tVals<-t(tVals)
head(tVals)

library(caret)
data("segmentationData")
cellData<-subset(segmentationData,Case=="Train")
cellData$Case<-cellData$Cell<-NULL
head(names(cellData))

rocValues<-filterVarImp(x=cellData[,-1],
                        y=cellData$Class)
head(rocValues)

library(CORElearn)
reliefValues<-attrEval(Class~.,data=cellData,
                       estimator="ReliefFequalK",
                       ReliefIterations=50)
head(reliefValues)

perm<-permuteRelief(x=cellData[,-1],
                    y=cellData$Class,
                    nperm=500,
                    estimator="ReliefFequalK",
                    ReliefIterations=50)

head(perm$permutations)

histogram(~value/Predictor,
          data=perm$permutations)
head(perm$standardized)

micValues<-mine(x=cellData[,-1],
                y=ifelse(cellData$Class=="PS",1,0))
head(micValues$MIC)

Sp62BTable<-table(training[pre2008,"Sponsor62B"],
                  training[pre2008,"Class"])

Sp62BTable
fisher.test(Sp62BTable)

ciTable<-table(training[pre2008,"CI.1950"],
               training[pre2008,"Class"])
ciTable
fisher.test(ciTable)

DayTable<-table(training[pre1008,"Weekday"],
           training[pre2008,"Class"])
DayTable
chisq.test(DayTable)

library(randomForest)
set.seed(791)
rfImp<-randomForest(Class~.,data=segTrain,
                    ntree=2000,
                    importance=TRUE)
head(varImp(rfImp))

library(AppliedPredictiveModeling)
data("abalone")
str(abalone)
head(abalone)
