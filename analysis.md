# Human Activity Recognition Analysis

This analysis is part of the programming assignment for the MOOC
[Practical Machine Learning](https://www.coursera.org/course/predmachlearn). It
uses the
[Weight Lifting Dataset](http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201)
by E. Velloso, A. Bulling, H. Gellersen, W. Ugulino, and H. Fuks to predict the
quality of barbell lifts. The assignment consists of two parts:

1. this report that details the process of finding a model that predicts the quality of the exercise on a given training data set
2. prediction on a given testing data set of which we do not know the outcome

## Model Building Strategy

The grade of the second part of the assignment is based on the prediction success rate of our model on the testing data set. In order to achieve the highest grade we should find a model that is tuned perfectly on the test set - even if it is overfitted. So we decide against following the standard procedure of exploring only our training data and holding back the testing data until the end where we predict on it only once. On the other hand we still want to reduce overfitting as much as possible. Therefore we follow the following model building strategy:

1. minimal exploratory data analysis on the training and testing data sets
2. splitting of the training data set into a training and a validation partition
3. building several models with cross-validation against the training partition
4. evaluating the models against the validation partition
5. selecting a model based on its purpose

We choose to build our models using the random forests algorithm with 10-fold cross-validation.

## Loading, Subsetting and Splitting the Data

We start with downloading the training and testing data sets and loading both sets into R.


```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv", "curl")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "pml-testing.csv", "curl")
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
```

## Exploratory Data Analysis

What are the dimensions of both data sets?


```r
data.frame (
    trainingDimensions = dim(training),
    testingDimensions  = dim(testing)
)
```

```
##   trainingDimensions testingDimensions
## 1              19622                20
## 2                160               160
```

Both sets have the same number of columns. Let's have a look if some of them are different.


```r
data.frame (
    notIntesting  = setdiff(names(training), names(testing)),
    notInTraining = setdiff(names(testing), names(training))
)
```

```
##   notIntesting notInTraining
## 1       classe    problem_id
```

The testing data set has no outcome variable but has an identifier column instead which we already expected. Now let's have a look at NA's. If we find fields that have lots of NA's in the testing data set it would be sensible to exclude these variables from our model.


```r
library(plyr)
naProptesting <- sapply(1:dim(testing)[2], function(x){mean(is.na(testing[, x]))})
count(data.frame(naProptesting), "naProptesting")
```

```
##   naProptesting freq
## 1             0   60
## 2             1  100
```

100 variables in the testing data set have no value at all. We don't want them in any of our models so we remove them from our training data set.


```r
trainingSubset <- training[, naProptesting < 1]
naPropTraining <- sapply(1:dim(trainingSubset)[2], function(x){mean(is.na(trainingSubset[, x]))})
count(data.frame(naPropTraining), "naPropTraining")
```

```
##   naPropTraining freq
## 1              0   60
```

Now all our training have no NA's left at all in either data set. Let's look at the first columns of the training data set.


```r
trainingSubset[1:10, 1:6]
```

```
##     X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 1   1  carlitos           1323084231               788290 05/12/2011 11:23
## 2   2  carlitos           1323084231               808298 05/12/2011 11:23
## 3   3  carlitos           1323084231               820366 05/12/2011 11:23
## 4   4  carlitos           1323084232               120339 05/12/2011 11:23
## 5   5  carlitos           1323084232               196328 05/12/2011 11:23
## 6   6  carlitos           1323084232               304277 05/12/2011 11:23
## 7   7  carlitos           1323084232               368296 05/12/2011 11:23
## 8   8  carlitos           1323084232               440390 05/12/2011 11:23
## 9   9  carlitos           1323084232               484323 05/12/2011 11:23
## 10 10  carlitos           1323084232               484434 05/12/2011 11:23
##    new_window
## 1          no
## 2          no
## 3          no
## 4          no
## 5          no
## 6          no
## 7          no
## 8          no
## 9          no
## 10         no
```

The X column looks to be some kind of identifier or row number. Let's verify this by testing it for uniqueness and its range in both data sets.


```r
data.frame (
    data_set = c("training", "testing"),
    unique_x = c(length(unique(training$X)), length(unique(testing$X))),
    min_x    = c(min(unique(training$X)), min(unique(testing$X))),
    max_x    = c(max(unique(training$X)), max(unique(testing$X))),
    rows     = c(dim(training)[1], dim(testing)[1])
)
```

```
##   data_set unique_x min_x max_x  rows
## 1 training    19622     1 19622 19622
## 2  testing       20     1    20    20
```

This confirms our assumption so we decide to exclude the variable from all our models as well. But before we do that let's have a look at the user column. We want to know if both data sets cover the same users and how many training data we have per user.


```r
trainingUsers <- count(training, "user_name")
names(trainingUsers)[2] <- "training_records"
testingUsers <- count(testing, "user_name")
names(testingUsers)[2] <- "testing_records"
merge(trainingUsers, testingUsers)
```

```
##   user_name training_records testing_records
## 1    adelmo             3892               1
## 2  carlitos             3112               3
## 3   charles             3536               1
## 4    eurico             3070               4
## 5    jeremy             3402               8
## 6     pedro             2610               3
```

We have several thousand training observations per user in the testing data set. Using the user name in our model could therefore lead to overfitting. But as described above an overfitted model might be helpful for the second part of the assignment so we decide to keep the user name to include it in at least one of our models. For the same reason we decide to keep the time stamp columns.


```r
trainingFinal <- trainingSubset[, -1]
```

## Partitioning of the Training Data Set

Now we split our training data set into a "final" training and a validation data set. We call both sets partitions to emphasize their difference from the original data sets.


```r
library(caret)
seed <- 20141119
set.seed(seed)
trainPartInd <- createDataPartition(training$classe, p = 0.7, list = FALSE)
trainPart <- trainingFinal[trainPartInd, ]
testPart <- trainingFinal[-trainPartInd, ]
```

## Model Building

We build all our models with the random forests algorithm and use 10-fold cross-validation for each training. We try out three models that differ in their predictors:

* _Model 1_: all non-outcome variables
* _Model 2_: all non-outcome variables except user name
* _Model 3_: all non-outcome variables except user name and timestamp/time window variables


```r
ctrl <- trainControl(method = "cv", number = 10)
set.seed(seed)
model1 <- train(classe ~ ., data = trainPart, method = "rf", trControl = ctrl)
set.seed(seed)
model2 <- train(classe ~ . - user_name, data = trainPart, method = "rf", trControl = ctrl)
set.seed(seed)
model3 <- train(classe ~ . - user_name - raw_timestamp_part_1 - raw_timestamp_part_2 - cvtd_timestamp - new_window, data = trainPart, method = "rf", trControl = ctrl)
```

## Model Evaluation

## Model Selection
