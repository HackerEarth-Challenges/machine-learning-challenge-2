path <- "/home/manish/Desktop/ML_Challenge_Creatives/Challenge #2 Data/kickstarter/comp_data/final_data/download_data/3149def2-5-datafiles/"
setwd(path)

# load data and libraries

library(data.table)
library(lubridate)
library(stringr)

train <- fread("train.csv")
test <- fread("test.csv")

# data dimension

sprintf("There are %s rows and %s columns in train data ",nrow(train),ncol(train))
sprintf("There are %s rows and %s columns in test data ",nrow(test),ncol(test))

# convert unix time format 

unix_feats <- c('deadline','state_changed_at','created_at','launched_at')
train[,c(unix_feats) := lapply(.SD, function(x) structure(x, class=c('POSIXct'))), .SDcols = unix_feats]
test[,c(unix_feats) := lapply(.SD, function(x) structure(x, class=c('POSIXct'))), .SDcols = unix_feats]

# create simple features

len_feats <- c('name_len','desc_len','keywords_len')
count_feats <- c('name_count','desc_count','keywords_count')
cols <- c('name','desc','keywords')

train[,c(len_feats) := lapply(.SD, function(x) str_count(x)), .SDcols = cols]
train[,c(count_feats) := lapply(.SD, function(x) str_count(x,"\\w+")), .SDcols = cols]

test[,c(len_feats) := lapply(.SD, function(x) str_count(x)), .SDcols = cols]
test[,c(count_feats) := lapply(.SD, function(x) str_count(x,"\\w+")), .SDcols = cols]

# encode features

train[,disable_communication := as.integer(as.factor(disable_communication))-1]
train[,country := as.integer(as.factor(country))-1]

test[,disable_communication := as.integer(as.factor(disable_communication))-1]
test[,country := as.integer(as.factor(country))-1]


# cols to use in modeling
cols_to_use <- c('final_status'
                 ,'name_len'
                 ,'desc_len'
                 ,'keywords_len'
                 ,'name_count'
                 ,'desc_count'
                 ,'keywords_count')


# GBM
library(gbm)
set.seed(1)

X_train <- copy(train)
X_train[,final_status := as.factor(final_status)]

clf_model <- gbm(final_status ~ .
                 ,data = train[,cols_to_use,with=F]
                 ,n.trees = 500
                 ,interaction.depth = 5
                 ,shrinkage = 0.3
                 ,train.fraction = 0.6
                 ,verbose = T)


# check variable importance
summary(clf_model, n.trees = 125)

# make predictions
clf_pred <- predict(clf_model, newdata = test, n.trees = 232,type = 'response')
clf_pred <- ifelse(clf_pred > 0.6,1,0)

# write file
subst <- data.table(project_id = test$project_id, final_status = clf_pred)
fwrite(subst, "gbm_starter.csv") #0.65754








