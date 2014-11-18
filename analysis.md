---
title: "Human Activity Recognition Analysis"
author: "Sebastian Bolz"
date: "18. November 2014"
output: html_document
---

This analysis is part of the programming assignment for the MOOC
[Practical Machine Learning](https://www.coursera.org/course/predmachlearn). It
uses the
[Weight Lifting Dataset](http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201)
by E. Velloso, A. Bulling, H. Gellersen, W. Ugulino, and H. Fuks to predict the
quality of barbell lifts.

## Loading, Subsetting and Splitting the Data

We start with downloading the training and testing data sets and loading both sets into R. We rename the testing data set into submission data set as we will later split off a testing data set from our training data set with which we estimate the out of sample error rate.


```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv", "curl")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-submission.csv", "pml-submission.csv", "curl")
training <- read.csv("pml-training.csv")
submission <- read.csv("pml-testing.csv")
```

What are the dimensions of both data sets?


```r
data.frame (
    trainingDimensions = dim(training),
    submissionDimensions  = dim(submission)
)
```

```
##   trainingDimensions submissionDimensions
## 1              19622                   20
## 2                160                  160
```

Both data sets have the same number of columns. Let's have a look if some of them are different.


```r
data.frame (
    notInsubmission  = setdiff(names(training), names(submission)),
    notInTraining = setdiff(names(submission), names(training))
)
```

```
##   notInsubmission notInTraining
## 1          classe    problem_id
```

The submission data set has no outcome variable but has an identifier column instead. This was expected and does not offer us any possibility to reduce the predictors from the training set. Now let's have a look at NA's. If we find fields that have lots of NA's in the submission data set it would be sensible to exclude these variables from our model.


```r
library(plyr)
naPropSubmission <- sapply(1:dim(submission)[2], function(x){mean(is.na(submission[, x]))})
count(data.frame(naPropSubmission), "naPropSubmission")
```

```
##   naPropSubmission freq
## 1                0   60
## 2                1  100
```

100 variables in the submission data set have no value at all. We don't want them in our model so we remove them from our training data set.


```r
trainingSubset <- training[, naPropSubmission < 1]
naPropTraining <- sapply(1:dim(trainingSubset)[2], function(x){mean(is.na(trainingSubset[, x]))})
count(data.frame(naPropTraining), "naPropTraining")
```

```
##   naPropTraining freq
## 1              0   60
```

Now all our training and submission variables have no NA's left at all. Let's look at the first columns of the submission data set.


```r
submission[1:6]
```

```
##     X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 1   1     pedro           1323095002               868349 05/12/2011 14:23
## 2   2    jeremy           1322673067               778725 30/11/2011 17:11
## 3   3    jeremy           1322673075               342967 30/11/2011 17:11
## 4   4    adelmo           1322832789               560311 02/12/2011 13:33
## 5   5    eurico           1322489635               814776 28/11/2011 14:13
## 6   6    jeremy           1322673149               510661 30/11/2011 17:12
## 7   7    jeremy           1322673128               766645 30/11/2011 17:12
## 8   8    jeremy           1322673076                54671 30/11/2011 17:11
## 9   9  carlitos           1323084240               916313 05/12/2011 11:24
## 10 10   charles           1322837822               384285 02/12/2011 14:57
## 11 11  carlitos           1323084277                36553 05/12/2011 11:24
## 12 12    jeremy           1322673101               442731 30/11/2011 17:11
## 13 13    eurico           1322489661               298656 28/11/2011 14:14
## 14 14    jeremy           1322673043               178652 30/11/2011 17:10
## 15 15    jeremy           1322673156               550750 30/11/2011 17:12
## 16 16    eurico           1322489713               706637 28/11/2011 14:15
## 17 17     pedro           1323094971               920315 05/12/2011 14:22
## 18 18  carlitos           1323084285               176314 05/12/2011 11:24
## 19 19     pedro           1323094999               828379 05/12/2011 14:23
## 20 20    eurico           1322489658               106658 28/11/2011 14:14
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
## 11         no
## 12         no
## 13         no
## 14         no
## 15         no
## 16         no
## 17         no
## 18         no
## 19         no
## 20         no
```

The goal of the assignment is to predict the outcome with data from the accelerometers which is why we will exclude the timestamp and time window information from our model. The X column looks to be some kind of identifier or row number. Let's verify this by testing it for uniqueness and its range in both data sets.


```r
data.frame (
    data_set = c("training", "submission"),
    unique_x = c(length(unique(training$X)), length(unique(submission$X))),
    min_x    = c(min(unique(training$X)), min(unique(submission$X))),
    max_x    = c(max(unique(training$X)), max(unique(submission$X))),
    rows     = c(dim(training)[1], dim(submission)[1])
)
```

```
##     data_set unique_x min_x max_x  rows
## 1   training    19622     1 19622 19622
## 2 submission       20     1    20    20
```

This proves it and makes us exclude the variable from our model as well. Now let's have a look at the user column. We want to if both data sets cover the same users and how many training data we have per user.


```r
trainingUsers <- count(training, "user_name")
names(trainingUsers)[2] <- "training_records"
submissionUsers <- count(submission, "user_name")
names(submissionUsers)[2] <- "submission_records"
merge(trainingUsers, submissionUsers)
```

```
##   user_name training_records submission_records
## 1    adelmo             3892                  1
## 2  carlitos             3112                  3
## 3   charles             3536                  1
## 4    eurico             3070                  4
## 5    jeremy             3402                  8
## 6     pedro             2610                  3
```

We have several thousand training observations per user in the submission data set. This enables us to build a separate model per user.


```r
library(caret)
```
