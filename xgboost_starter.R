
# This script will help you learn how to build a xgboost models on features extracted using 
# Text Mining methods. This script scores ~0.70 on public leaderboard.


# Load Libraries ----------------------------------------------------------

library(data.table)
library(stringr)
library(text2vec)

train <- fread("train.csv")
test <- fread("test.csv")


# Convert Unix Time Format ------------------------------------------------

unix_feats <- c('deadline','state_changed_at','created_at','launched_at')
train[,c(unix_feats) := lapply(.SD, function(x) structure(x, class=c('POSIXct'))), .SDcols = unix_feats]
test[,c(unix_feats) := lapply(.SD, function(x) structure(x, class=c('POSIXct'))), .SDcols = unix_feats]


# Create Features ---------------------------------------------------------

len_feats <- c('name_len','desc_len','keywords_len')
count_feats <- c('name_count','desc_count','keywords_count')
cols <- c('name','desc','keywords')

train[,c(len_feats) := lapply(.SD, function(x) str_count(x)), .SDcols = cols]
train[,c(count_feats) := lapply(.SD, function(x) str_count(x,"\\w+")), .SDcols = cols]

test[,c(len_feats) := lapply(.SD, function(x) str_count(x)), .SDcols = cols]
test[,c(count_feats) := lapply(.SD, function(x) str_count(x,"\\w+")), .SDcols = cols]


# Some More Features ------------------------------------------------------

train[,time1 := as.numeric(difftime(launched_at, created_at))]
train[,time3 := as.numeric(difftime(deadline, launched_at))]

train[,time1 := log(time1)]
train[,time3 := log(time3)]

test[,time1 := as.numeric(difftime(launched_at, created_at))]
test[,time3 := as.numeric(difftime(deadline, launched_at))]

test[,time1 := log(time1)]
test[,time3 := log(time3)]



# Encoding Variables ------------------------------------------------------

train[,disable_communication := as.integer(as.factor(disable_communication))-1]
test[,disable_communication := as.integer(as.factor(disable_communication))-1]

countryall <- data.table(country = append(train$country, test$country))
countryall[,country := as.integer(as.factor(country))-1]

country_train <- countryall[1:nrow(train)]
country_test <- countryall[(nrow(train)+1):nrow(countryall)]

train[,country := NULL][,country := country_train$country]
test[,country := NULL][, country := country_test$country]

train[,goal := log1p(goal)]
test[,goal := log1p(goal)]

rm(country_test,country_train,countryall)
gc()



# Creating Features from 'Keywords' Variable ------------------------------

# We could have use a R package to perform the following text mining steps.
# Rather we'll follow a manual cleaning process which will help you learn using regular expressions as well

#creating a data frame by combining keywords from both data sets
fullkey <- rbind(train[,.(project_id,keywords)], test[,.(project_id, keywords)])



# Text Cleaning -----------------------------------------------------------

fullkey[,keywords := lapply(keywords, function(x) str_split(string = x, pattern = "-"))]

# function to remove stop words
remov_stop <- function(x){
  
  t <- unlist(x)
  t <- setdiff(t, tidytext::stop_words$word)
  return (t)
  
}

fullkey[,keywords := lapply(keywords, function(x) remov_stop(x))]
fullkey[,keywords := lapply(keywords, function(x) str_replace_all(x, "[[:digit:]]",""))]
fullkey[,keywords := lapply(keywords, function(x) SnowballC::wordStem(x))]
fullkey[, keywords := lapply(keywords, function(x) x[nchar(x) > 2])]


# creating count corpus

vec_train <- itoken(fullkey$keywords,tokenizer = word_tokenizer,ids = fullkey$project_id)
vocab = create_vocabulary(vec_train)
vocab

pruned_vocab <- prune_vocabulary(vocab,term_count_min = 150) # words occuring 150 or more times
pruned_vocab

vocab1 <- vocab_vectorizer(pruned_vocab)
dtm_text <- create_dtm(vec_train,vocab1)
dim(dtm_text)

dtm_text1 <- as.data.table(as.matrix(dtm_text))

dtm_train <- dtm_text1[1:108129]
dtm_test <- dtm_text1[108130:171594]


# Adding text features in train and test data -----------------------------

X_train <- copy(train)
X_test <- copy(test)

cols_to_use <- c('name_len'
                 ,'desc_len'
                 ,'keywords_len'
                 ,'name_count'
                 ,'desc_count'
                 ,'keywords_count'
                 ,'time1'
                 ,'time3'
                 ,'goal')

X_train <- cbind(X_train[,cols_to_use,with=F], dtm_train)
X_test <- cbind(X_test[,cols_to_use,with=F], dtm_test)

X_train <- cbind(X_train, train_isnum$is_number)
X_test <- cbind(X_train, test_isnum$is_number)


# Model Training ----------------------------------------------------------

library(xgboost)

dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = as.numeric(train$final_status))
dtest <- xgb.DMatrix(data = as.matrix(X_test))

params <- list(
  
  objective = "binary:logistic",
  eta = 0.025,
  max_depth = 6,
  subsample = 0.7,
  colsample_bytree = 0.7,
  min_child_weight = 5
  
)

big_cv <- xgb.cv(params = params
                 ,data = dtrain
                 ,nrounds = 1000
                 ,nfold = 5L
                 ,metrics = 'error'
                 ,stratified = T
                 ,print_every_n = 10
                 ,early_stopping_rounds = 40)

iter <- big_cv$best_iteration

big_train <- xgb.train(params = params
                       ,data = dtrain
                       ,nrounds = iter)

imp <- xgb.importance(model = big_train, feature_names = colnames(dtrain))
xgb.plot.importance(imp,top_n = 20)

big_pred <- predict(big_train, dtest)
big_pred <- ifelse(big_pred > 0.5,1,0)

sub <- data.table(project_id = test$project_id, final_status = big_pred)
fwrite(sub, "xgb_with_feats.csv") #0.703








