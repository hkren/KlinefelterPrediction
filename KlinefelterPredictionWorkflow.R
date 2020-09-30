#####################
#required r packages#
#####################
library(data.table)
library(e1071)
library(GenSA)
library(glmnet)
library(kknn)
library(MASS)
library(mlr3)
library(mlr3learners)
library(mlr3measures)
library(mlr3tuning)
library(paradox)
library(ranger)
library(stats)
library(xgboost)

###########
#functions#
###########

#bootstrap list of training and validation sets
#group -> index of observations in group
#iter -> number of bootstrap samples to be created
#no -> number of observations in both groups
createBootstrap<-function(group,iter=1,no){
  trainSets<-list()
  testSets<-list()
  for(i in 1:iter){  
    sampling<-sample(group,no,replace=TRUE)
    trainSets<-append(trainSets,list(sampling))
    testSets<-append(testSets,list(setdiff(group,sampling)))
  }
  return(list("trainSets"=trainSets,"testSets"=testSets))
}

#bootstrap list of equally sized training and validation sets for two groups
#groupA -> index of observations in first group
#groupB -> index of observations in second group
#iter -> number of bootstrap samples to be created
createBootstrapOfTwoGroups<-function(groupA,groupB,iter=1){
  bootstrapA<-createBootstrap(groupA,iter,(length(groupA)+length(groupB))/2)
  bootstrapB<-createBootstrap(groupB,iter,(length(groupA)+length(groupB))/2)
  trainSets<-list()
  testSets<-list()
  for(i in 1:iter){
    trainSets<-append(trainSets,list(c(unlist(bootstrapA[[1]][i]),unlist(bootstrapB[[1]][i]))))
    testSets<-append(testSets,list(c(unlist(bootstrapA[[2]][i]),unlist(bootstrapB[[2]][i]))))
  }
  return(list("trainSets"=trainSets,"testSets"=testSets))
}

#Evaluation of model performances
#Calculate sensitivity, specificity and Youden Index for all models and all thresholds in 0.01 to 0.99
#Calculate AUC of all models
#resamplingResults -> list of all models (object as returned from MLR3)
modelEvalutation<-function(resamplingResults){
  
  modelEval<-data.table("learner"=character(),"model"=numeric(),"threshold"=numeric(),
             "sensitivity"=numeric(),"specificity"=numeric(),"youdenIndex"=numeric(),"ce"=numeric())
  
  modelAuc<-data.table("learner"=character(),"model"=numeric(),"auc"=numeric())
  
  #each sML algorithm creates one resamplingResult
  for(resamplingResult in resamplingResults){
    
    #number of predictions equals number of bootstrap sets for training and validation
    for(i in 1:length(resamplingResult$predictions())){
      
      resultTable<-as.data.table(resamplingResult$predictions()[[i]])
      
      #get correct predictions and measures for each threshold between 0.01 and 0.99
      for(j in seq(0.01,0.99,by=0.01)){
        thresholdX<-resultTable$prob.1>j
        modelEval<-rbind(modelEval,list(resamplingResult$learners[[1]]$id,i,j,
                                        #sensitivity
                                        sum(thresholdX==TRUE&resultTable$truth==1)/sum(resultTable$truth==1),
                                        #specificity
                                        sum(thresholdX==FALSE&resultTable$truth==0)/sum(resultTable$truth==0),
                                        #youdenIndex
                                        sum(thresholdX==TRUE&resultTable$truth==1)/sum(resultTable$truth==1)+
                                          sum(thresholdX==FALSE&resultTable$truth==0)/sum(resultTable$truth==0)-1,
                                        #classification error
                                        (sum(thresholdX==TRUE&resultTable$truth==0)+sum(thresholdX==FALSE&resultTable$truth==1))/
                                          length(thresholdX)))
      }
      #get AUC
      modelAuc<-rbind(modelAuc,list(resamplingResult$learners[[1]]$id,i,mlr3measures::auc(resultTable$truth,resultTable$prob.1,"1")))
    }
  }
  return(list(modelEval,modelAuc))
}

#Evaluation of model performances for feature importance
#Calculate AUC of all models
#resamplingResults -> list of all models (object as returned from MLR3)
modelEvalutationFeatureImportance<-function(resamplingResults){
  
  modelAuc<-data.table("learner"=character(),"task"=numeric(),"model"=numeric(),"auc"=numeric())
  
  for(resamplingResult in resamplingResults){
    for(i in 1:length(resamplingResult$predictions())){
      resultTable<-as.data.table(resamplingResult$predictions()[[i]])
      modelAuc<-rbind(modelAuc,list(resamplingResult$learners[[1]]$id,resamplingResult$task$id,i,mlr3measures::auc(resultTable$truth,resultTable$prob.1,"1")))
    }
  }
  return(modelAuc)
}

######
#Data#
######

#seed for the random processes in this section
set.seed(9238)

#read in data
retrospectiveData <- fread(file="retrospectiveData.csv",sep=",",header = TRUE)

prospectiveDataAzoo <- fread(file="prospectiveDataAzoo.csv",sep=",",header = TRUE)

prospectiveDataCrypto <- fread(file="prospectiveDataCrypto.csv",sep=",",header = TRUE)


#keep relevant features
#KS coded as factor: 0 control, 1 KS
retrospectiveData <- retrospectiveData[,list(age,height,BMI,testisVolumne,FSH,LH,testosterone,prolactin,estradiol,pH,KS)]
prospectiveDataAzoo <- prospectiveDataAzoo[,list(age,height,BMI,testisVolumne,FSH,LH,testosterone,prolactin,estradiol,pH,KS)]
prospectiveDataCrypto <- prospectiveDataCrypto[,list(age,height,BMI,testisVolumne,FSH,LH,testosterone,prolactin,estradiol,pH,KS)]
retrospectiveData$KS <- as.factor(retrospectiveData$KS)
prospectiveDataAzoo$KS <- as.factor(prospectiveDataAzoo$KS)
prospectiveDataCrypto$KS <- as.factor(prospectiveDataCrypto$KS)

#split retrospectiveData in test and training/validation data
#20% of KS and control patients for test
testIndex<-c(sample(row.names(retrospectiveData)[retrospectiveData$KS==0],
                    0.2*sum(retrospectiveData$KS==0),replace = FALSE),
             sample(row.names(retrospectiveData)[retrospectiveData$KS==1],
                    0.2*sum(retrospectiveData$KS==1),replace=FALSE))
testData<-retrospectiveData[row.names(retrospectiveData)%in%testIndex,]

#remaining 80% of observations for training/validation
valTrainIndex<-setdiff(row.names(retrospectiveData), testIndex)
valTrainData<-retrospectiveData[row.names(retrospectiveData)%in%valTrainIndex,]

##############################
#Descriptive analysis of data#
##############################

#median and ranges
stats<-data.table("feature"=numeric(),"medienAll"=numeric(),"minAll"=numeric(),"maxAll"=numeric(),
                  "medianControl"=numeric(),"minControl"=numeric(),"maxControl"=numeric(),
                  "medianKS"=numeric(),"minKS"=numeric(),"maxKS"=numeric(),
                  "wilcox"=numeric())

for(i in c("age","height","BMI","testisVolumne","FSH","LH","testosterone","prolactin","estradiol","pH")){
  stats<-rbind(stats,list(i,
                          round(median(retrospectiveData[,get(i)]),2),
                          round(min(retrospectiveData[,get(i)]),2),
                          round(max(retrospectiveData[,get(i)]),2),
                          round(median(retrospectiveData[retrospectiveData$KS==0,get(i)]),2),
                          round(min(retrospectiveData[retrospectiveData$KS==0,get(i)]),2),
                          round(max(retrospectiveData[retrospectiveData$KS==0,get(i)]),2),
                          round(median(retrospectiveData[retrospectiveData$KS==1,get(i)]),2),
                          round(min(retrospectiveData[retrospectiveData$KS==1,get(i)]),2),
                          round(max(retrospectiveData[retrospectiveData$KS==1,get(i)]),2),
                          wilcox.test(retrospectiveData[retrospectiveData$KS==0,get(i)],retrospectiveData[retrospectiveData$KS==1,get(i)],paired = FALSE)$p.value))
}

#Principle component analysis
pca <- prcomp(retrospectiveData[,list(age,height,BMI,testisVolumne,FSH,LH,testosterone,prolactin,estradiol,pH)],center = TRUE,scale=TRUE)
summary(pca)

#correlations between features (test spearman, kendall and pearson)
abs(cor(as.matrix(retrospectiveData[,list(age,height,BMI,testisVolumne,FSH,LH,testosterone,prolactin,estradiol,pH)]),method="spearman"))>=0.3
abs(cor(as.matrix(retrospectiveData[,list(age,height,BMI,testisVolumne,FSH,LH,testosterone,prolactin,estradiol,pH)]),method="kendall"))>=0.3
abs(cor(as.matrix(retrospectiveData[,list(age,height,BMI,testisVolumne,FSH,LH,testosterone,prolactin,estradiol,pH)]),method="pearson"))>=0.3

###################################
#MLR3 Objects for machine learning#
###################################

#seed for the random processes in this section
set.seed(4208)

#mlr3 task: data -> training/validation data
task = TaskClassif$new(id = "task", backend = valTrainData, target = "KS", positive="1")

#create 20 bootstrap sets from training/validation data (draw with replacement)
#bootstrap set: distinct training and validation sets, #training set = #training/validation data
bootstrapList<-createBootstrapOfTwoGroups(task$row_ids[task$data(cols="KS")==1],
                                          task$row_ids[task$data(cols="KS")==0],20)

#mlr3 resampling (multiple train and validation runs): custom resampling for bootstrap sets
resampling = rsmp("custom")
resampling$instantiate(task,bootstrapList[[1]],bootstrapList[[2]])

#######################
#Hyperparameter tuning#
#######################

#mlr3 measures for model evaluation
measures = list(msr("classif.ce"),msr("classif.auc"),msr("classif.sensitivity"),msr("classif.specificity"))

#mlr3 number of evaluations for each model and parameter setting
evals = term("evals", n_evals = 100)

#seed for the random processes in this section
set.seed(2735)

#Algorithm: rpart
#Parameters to tune: cp, minsplit, maxdepth
hyperLearner = lrn("classif.rpart")
hyperLearner$predict_type="prob"
tune_ps = ParamSet$new(list(
  ParamDbl$new("cp", lower = 0.001, upper = 1),
  ParamInt$new("minsplit", lower = 1, upper = 100),
  ParamInt$new("maxdepth",lower = 1, upper = 30)
))

instance = TuningInstance$new(
  task = task,
  learner = hyperLearner,
  resampling = resampling,
  measures = measures,
  param_set = tune_ps,
  terminator = evals
)

tuner = tnr("random_search")
result = tuner$tune(instance)
result_figuresRpart<-as.data.table(instance$archive(unnest = "params")[, c("cp", "minsplit","maxdepth", "classif.ce","classif.auc","classif.specificity","classif.sensitivity")])

#seed for the random processes in this section
set.seed(2735)

#Algorithm: ranger
#Parameters to tune: num.trees, min.node.size, importance
hyperLearner = lrn("classif.ranger")
hyperLearner$predict_type="prob"
tune_psGini = ParamSet$new(list(
  ParamFct$new("splitrule", levels = c("gini")),
  ParamInt$new("num.trees", lower = 100, upper = 1000),
  ParamInt$new("min.node.size",lower = 1, upper = 100),
  ParamFct$new("importance", levels = c("none","impurity","permutation"))
))

tune_psExtratrees = ParamSet$new(list(
  ParamFct$new("splitrule", levels = c("extratrees")),
  ParamInt$new("num.trees", lower = 100, upper = 1000),
  ParamInt$new("min.node.size",lower = 1, upper = 100),
  ParamInt$new("num.random.splits", lower= 1, upper=10),
  ParamFct$new("importance", levels = c("none","impurity","permutation"))
))

instance = TuningInstance$new(
  task = task,
  learner = hyperLearner,
  resampling = resampling,
  measures = measures,
  param_set = tune_psGini,
  terminator = evals
)

tuner = tnr("random_search")
result = tuner$tune(instance)
result_figuresGini<-as.data.table(instance$archive(unnest = "params")[, c("splitrule", "num.trees","min.node.size","importance", "classif.ce","classif.auc","classif.specificity","classif.sensitivity")])

instance = TuningInstance$new(
  task = task,
  learner = hyperLearner,
  resampling = resampling,
  measures = measures,
  param_set = tune_psExtratrees,
  terminator = evals
)

tuner = tnr("random_search")
result = tuner$tune(instance)

result_figuresExtratrees<-as.data.table(instance$archive(unnest = "params")[, c("splitrule", "num.trees","min.node.size","importance","num.random.splits", "classif.ce","classif.auc","classif.specificity","classif.sensitivity")])

#seed for the random processes in this section
set.seed(2735)

#Algorithm: lda
#Parameters to tune: method, predict.method
hyperLearner = lrn("classif.lda")
hyperLearner$predict_type="prob"

hyperLearner$param_set
tune_ps = ParamSet$new(list(
  ParamFct$new("method",levels = c("moment", "mle",    "mve",    "t")),
  ParamFct$new("predict.method",levels = c("plug-in",   "predictive", "debiased" ))
))

instance = TuningInstance$new(
  task = task,
  learner = hyperLearner,
  resampling = resampling,
  measures = measures,
  param_set = tune_ps,
  terminator = evals
)

tuner = tnr("grid_search",resolution=4)
result = tuner$tune(instance)
result_figuresLda<-as.data.table(instance$archive(unnest = "params")[, c("method", "predict.method", "classif.ce","classif.auc","classif.specificity","classif.sensitivity")])

#seed for the random processes in this section
set.seed(2735)

#Algorithm: log_reg
#Parameters to tune: epsilon, maxit
hyperLearner = lrn("classif.log_reg")
hyperLearner$predict_type="prob"

hyperLearner$param_set$default
tune_ps = ParamSet$new(list(
  ParamDbl$new("epsilon",lower=0,upper=1),
  ParamDbl$new("maxit", lower=0, upper=100)
))

instance = TuningInstance$new(
  task = task,
  learner = hyperLearner,
  resampling = resampling,
  measures = measures,
  param_set = tune_ps,
  terminator = evals
)

tuner = tnr("random_search")
result = tuner$tune(instance)
result_figuresLog<-as.data.table(instance$archive(unnest = "params")[, c("epsilon", "maxit", "classif.ce","classif.auc","classif.specificity","classif.sensitivity")])

#seed for the random processes in this section
set.seed(2735)

#Algorithm: qda
#Parameters to tune: method, predict.method
hyperLearner = lrn("classif.qda")
hyperLearner$predict_type="prob"

hyperLearner$param_set$default
tune_ps = ParamSet$new(list(
  ParamFct$new("method",levels = c("moment", "mle",    "mve",    "t")),
  ParamFct$new("predict.method",levels = c("plug-in",   "predictive", "debiased" ))
))

instance = TuningInstance$new(
  task = task,
  learner = hyperLearner,
  resampling = resampling,
  measures = measures,
  param_set = tune_ps,
  terminator = evals
)

tuner = tnr("grid_search",resolution=4)
result = tuner$tune(instance)
result_figuresQda<-as.data.table(instance$archive(unnest = "params")[, c("method", "predict.method", "classif.ce","classif.auc","classif.specificity","classif.sensitivity")])

#seed for the random processes in this section
set.seed(2735)

#Algorithm: svm
#Parameters to tune: type, kernel,cost,degree
hyperLearner = lrn("classif.svm")
hyperLearner$predict_type="prob"

hyperLearner$param_set$default
tune_ps = ParamSet$new(list(
  ParamFct$new("type",levels = c("C-classification")),
  ParamFct$new("kernel",levels = c("linear", "radial", "sigmoid")),
  ParamDbl$new("cost",lower=0,upper=100)
))

tune_psPoly = ParamSet$new(list(
  ParamFct$new("type",levels = c("C-classification")),
  ParamFct$new("kernel",levels = c("polynomial")),
  ParamInt$new("degree",lower=2,upper=10),
  ParamDbl$new("cost",lower=0,upper=100)
))

instance = TuningInstance$new(
  task = task,
  learner = hyperLearner,
  resampling = resampling,
  measures = measures,
  param_set = tune_ps,
  terminator = evals
)

instancePoly = TuningInstance$new(
  task = task,
  learner = hyperLearner,
  resampling = resampling,
  measures = measures,
  param_set = tune_psPoly,
  terminator = evals
)

tuner = tnr("random_search")
result = tuner$tune(instance)
result = tuner$tune(instancePoly)
result_figuresSvm<-as.data.table(instance$archive(unnest = "params")[, c("kernel", "cost", "classif.ce","classif.auc","classif.specificity","classif.sensitivity")])
result_figuresSvmPoly<-as.data.table(instancePoly$archive(unnest = "params")[, c("kernel", "cost","degree", "classif.ce","classif.auc","classif.specificity","classif.sensitivity")])

#seed for the random processes in this section
set.seed(2735)

#Algorithm: glmnet
#Parameters to tune: alpha
hyperLearner = lrn("classif.glmnet")
hyperLearner$predict_type="prob"

tune_ps = ParamSet$new(list(
  ParamDbl$new("alpha",lower=0,upper=1)
))

instance = TuningInstance$new(
  task = task,
  learner = hyperLearner,
  resampling = resampling,
  measures = measures,
  param_set = tune_ps,
  terminator = evals
)

tuner = tnr("random_search")
result = tuner$tune(instance)
result_figuresGlmnet<-as.data.table(instance$archive(unnest = "params")[, c("alpha","classif.ce","classif.auc","classif.specificity","classif.sensitivity")])

#seed for the random processes in this section
set.seed(2735)

#Algorithm: xgboost
#Parameters to tune: nrounds,eta,gamma,max_depth,min_child_weight
hyperLearner = lrn("classif.xgboost")
hyperLearner$predict_type="prob"
tune_ps = ParamSet$new(list(
  ParamInt$new("nrounds",lower=100,upper=1000),
  ParamDbl$new("eta",lower=0.01,upper=0.03),
  ParamDbl$new("gamma",lower=0,upper=10),
  ParamInt$new("max_depth",lower=1,upper=30),
  ParamInt$new("min_child_weight",lower=1,upper=100)
))

instance = TuningInstance$new(
  task = task,
  learner = hyperLearner,
  resampling = resampling,
  measures = measures,
  param_set = tune_ps,
  terminator = evals
)

tuner = tnr("random_search")
result = tuner$tune(instance)
result_figuresXgboost<-as.data.table(instance$archive(unnest = "params")[, c("nrounds","eta","gamma","max_depth","min_child_weight","classif.ce","classif.auc","classif.specificity","classif.sensitivity")])

#seed for the random processes in this section
set.seed(2735)

#Algorithm: kknn
#Parameters to tune: kernel,k,distance
hyperLearner = lrn("classif.kknn")
hyperLearner$predict_type="prob"
tune_ps = ParamSet$new(list(
  ParamFct$new("kernel", levels=c("rectangular", "triangular", "epanechnikov", "biweight", "triweight", "cos", "inv", "gaussian",  "rank", "optimal" )),
  ParamInt$new("k", lower = 1, upper = 20),
  ParamInt$new("distance",lower = 1, upper = 1)
))

instance = TuningInstance$new(
  task = task,
  learner = hyperLearner,
  resampling = resampling,
  measures = measures,
  param_set = tune_ps,
  terminator = evals
)

tuner = tnr("random_search")
result = tuner$tune(instance)
result_figuresKknn<-as.data.table(instance$archive(unnest = "params")[, c("kernel", "k","distance", "classif.ce","classif.auc","classif.specificity","classif.sensitivity")])

#######################
#MLR3 Objects: learner#
#######################

learner <- lapply(c("classif.featureless","classif.glmnet","classif.lda","classif.log_reg","classif.kknn",
                    "classif.qda","classif.ranger","classif.rpart","classif.svm","classif.xgboost"), 
                  lrn, predict_sets = c("train", "test"), predict_type="prob")

#set hyperparameters according to tuning
learner[[2]]$param_set$values = list(alpha=result_figuresGlmnet[classif.auc==max(classif.auc),alpha][1])

learner[[3]]$param_set$values = list(method = result_figuresLda[classif.auc==max(classif.auc),method][1],
                                     predict.method = result_figuresLda[classif.auc==max(classif.auc),predict.method][1])

learner[[4]]$param_set$values = list(epsilon=result_figuresLog[classif.auc==max(classif.auc),epsilon][1],
                                     maxit=result_figuresLog[classif.auc==max(classif.auc),maxit][1])

learner[[5]]$param_set$values = list(k=result_figuresKknn[classif.auc==max(classif.auc),k][1],
                                     kernel=result_figuresKknn[classif.auc==max(classif.auc),kernel][1],
                                     distance=result_figuresKknn[classif.auc==max(classif.auc),distance][1])

learner[[6]]$param_set$values = list(method = result_figuresQda[classif.auc==max(classif.auc),method][1],
                                     predict.method = result_figuresQda[classif.auc==max(classif.auc),predict.method][1])

learner[[7]]$param_set$values = list(splitrule = result_figuresExtratrees[classif.auc==max(classif.auc),splitrule][1],
                                     num.trees = result_figuresExtratrees[classif.auc==max(classif.auc),num.trees][1],
                                     min.node.size = result_figuresExtratrees[classif.auc==max(classif.auc),min.node.size][1],
                                     importance = result_figuresExtratrees[classif.auc==max(classif.auc),importance][1],
                                     num.random.splits = result_figuresExtratrees[classif.auc==max(classif.auc),num.random.splits][1])

learner[[8]]$param_set$values = list(cp=result_figuresRpart[classif.auc==max(classif.auc),cp][1],
                                     minsplit=result_figuresRpart[classif.auc==max(classif.auc),minsplit][1],
                                     maxdepth=result_figuresRpart[classif.auc==max(classif.auc),maxdepth][1])

learner[[9]]$param_set$values = list(kernel=result_figuresSvm[classif.auc==max(classif.auc),kernel][1],
                                     type="C-classification",
                                     cost=result_figuresSvm[classif.auc==max(classif.auc),cost][1])

learner[[10]]$param_set$values = list(nrounds=result_figuresXgboost[classif.auc==max(classif.auc),nrounds][1],
                                      eta=result_figuresXgboost[classif.auc==max(classif.auc),eta][1],
                                      gamma=result_figuresXgboost[classif.auc==max(classif.auc),gamma][1],
                                      max_depth=result_figuresXgboost[classif.auc==max(classif.auc),max_depth][1],                                     
                                      min_child_weight=result_figuresXgboost[classif.auc==max(classif.auc),min_child_weight][1])

################
#Model training#
################

#seed for the random processes in this section
set.seed(6542)

resamplingResults <- lapply(learner,resample,task=task,resampling=resampling, store_models = TRUE)

#############################################
#Model evaluation (based on validation sets)#
#############################################

#seed for the random processes in this section
set.seed(3112)

#evaluate model performances (AUC, sensitivy, specificity, youden index, classification errer for all thresholds in range 0.01 to 0.99)
modelPerfomances<-modelEvalutation(resamplingResults)

#mean performances and variance of perfomances (sensitivity, specificity, youden index, classification error) per learner and threshold
modelPerfomancesMeanByThreshold<-modelPerfomances[[1]][,list(youdenIndex=mean(youdenIndex),youdenIndexVar=var(youdenIndex),
                                                    sensitivity=mean(sensitivity),sensitivityVar=var(sensitivity),
                                                    specificity=mean(specificity),specificityVar=var(specificity),
                                                    ce=mean(ce),ce=var(ce)), by = list(learner,threshold)]

#For each model type: Choose min threshold with sensitivity>=0.95
#prediction > threshold -> patient is suspected to have KS syndrome
modelBestThresholds<-modelPerfomancesMeanByThreshold[sensitivity>=0.95,list(threshold=which.max(specificity)/100,
                                specificity=max(specificity),sensitivity=min(sensitivity)),by =learner]

modelBestThresholdsAllParameters <- modelPerfomancesMeanYI[(1:10-1)*99+(modelBestThresholds$threshold[1:10]*100)]

#mean AUC per learner
modelAucMean<-modelPerfomances[[2]][,list(meanAUC=mean(auc),varAUC=var(auc),minAUC=min(auc),maxAUC=max(auc),which.max(auc)), by = learner]

####################
#Feature Importance#
####################

#seed for the random processes in this section
set.seed(7912)

#repeat model training but leave out each feature
resamplingResultsFeature<-list()

for(feature in 1:ncol(valTrainData)){
  
  if(colnames(valTrainData)[feature]!="KS"){
    
    myDataFeature <- valTrainData[,-feature, with=FALSE]
    taskFeature <- TaskClassif$new(id = colnames(valTrainData)[feature], 
                                   backend = myDataFeature, target = "KS", positive="1")
    resamplingFeature = rsmp("custom")
    resamplingFeature$instantiate(taskFeature,bootstrapList[[1]],bootstrapList[[2]])
    resamplingResultsFeature <- append(resamplingResultsFeature,lapply(learner,resample,task=taskFeature,resampling=resamplingFeature, store_models = FALSE))
    
  }
}

#evaluate model performances -> AUC
modelPerfomancesFeatureImportance<-modelEvalutationFeatureImportance(resamplingResultsFeature)

#mean AUC per learner and feature
modelAucMeanFeatureImportance<-modelPerfomancesFeatureImportance[,list(meanAUC=mean(auc),varAUC=var(auc),minAUC=min(auc),maxAUC=max(auc),which.max(auc)), by = list(learner,task)]

#calculate feature importance as auc with all features divided by auc without feature
modelAucMeanFeatureImportance <- merge(x = modelAucMeanFeatureImportance,y = modelAucMean[,list(learner,auc_allFeatures=meanAUC)],by = "learner")
modelAucMeanFeatureImportance$featureImportance <- modelAucMeanFeatureImportance$auc_allFeatures/modelAucMeanFeatureImportance$meanAUC

#######################################
#Model evaluation (based on test sets)#
#######################################

#seed for the random processes in this section
set.seed(3512)

taskTest = TaskClassif$new(id = "taskTest", backend = testData, target = "KS", positive="1")

testResults<-data.table("learner"=character(),"data"=character(),"threshold"=numeric(),
                                  "sensitivity"=numeric(),"specificity"=numeric(),"ce"=numeric(),
                                  "auc"=numeric())

#test the best model of each model type (retrospective test data)
for(resamplingResult in resamplingResults){
  
  if(resamplingResult$learners[[
      modelPerfomances[[2]][learner==resamplingResult$learners[[1]]$id,which.max(auc)]]]$id%in%modelBestThresholds$learner){
  
  #prediction of retrospective test data
  x<-resamplingResult$learners[[modelPerfomances[[2]][learner==resamplingResult$learners[[1]]$id,which.max(auc)]]]$predict_newdata(newdata = testData, task=taskTest)
  #set to best threshold
  x$set_threshold(modelBestThresholds[
    learner==resamplingResult$learners[[
      modelPerfomances[[2]][learner==resamplingResult$learners[[1]]$id,which.max(auc)]]]$id,threshold])
  #get quality measures
  testResults<-rbind(testResults,list(resamplingResult$learners[[1]]$id,"test",modelBestThresholds[learner==resamplingResult$learners[[1]]$id,threshold],x$score(msr("classif.sensitivity")),x$score(msr("classif.specificity")),x$score(msr("classif.ce")),x$score(msr("classif.auc"))))
  }
}

#test the best model of each model type (prospective test data (azoospermia))
for(resamplingResult in resamplingResults){
  
  if(resamplingResult$learners[[
    modelPerfomances[[2]][learner==resamplingResult$learners[[1]]$id,which.max(auc)]]]$id%in%modelBestThresholds$learner){
    
    #prediction of prospective test data (azoospermia)
    x<-resamplingResult$learners[[modelPerfomances[[2]][learner==resamplingResult$learners[[1]]$id,which.max(auc)]]]$predict_newdata(newdata = prospectiveDataAzoo, task=taskTest)
    #set to best threshold
    x$set_threshold(modelBestThresholds[
      learner==resamplingResult$learners[[
        modelPerfomances[[2]][learner==resamplingResult$learners[[1]]$id,which.max(auc)]]]$id,threshold])
    #get quality measures
    testResults<-rbind(testResults,list(resamplingResult$learners[[1]]$id,"test",modelBestThresholds[learner==resamplingResult$learners[[1]]$id,threshold],x$score(msr("classif.sensitivity")),x$score(msr("classif.specificity")),x$score(msr("classif.ce")),x$score(msr("classif.auc"))))
  }
}

#test the best model of each model type (prospective test data (cryptozoospermia))
for(resamplingResult in resamplingResults){
  
  if(resamplingResult$learners[[
    modelPerfomances[[2]][learner==resamplingResult$learners[[1]]$id,which.max(auc)]]]$id%in%modelBestThresholds$learner){
    
    #prediction of prospective test data (cryptozoospermia)
    x<-resamplingResult$learners[[modelPerfomances[[2]][learner==resamplingResult$learners[[1]]$id,which.max(auc)]]]$predict_newdata(newdata = prospectiveDataCrypto, task=taskTest)
    #set to best threshold
    x$set_threshold(modelBestThresholds[
      learner==resamplingResult$learners[[
        modelPerfomances[[2]][learner==resamplingResult$learners[[1]]$id,which.max(auc)]]]$id,threshold])
    #get quality measures
    testResults<-rbind(testResults,list(resamplingResult$learners[[1]]$id,"test",modelBestThresholds[learner==resamplingResult$learners[[1]]$id,threshold],x$score(msr("classif.sensitivity")),x$score(msr("classif.specificity")),x$score(msr("classif.ce")),x$score(msr("classif.auc"))))
  }
}
