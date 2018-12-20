library(readr) #to load in csv, txt files
install.packages("caret")
library(caret) # to get stratified train and test set
library(party) #for ctree


tictactoe <- read_csv("Desktop/UH/Fall2018/DM4335/Project3/tictactoe.txt", 
                      col_names = FALSE)
colnames(tictactoe) = c('top_left', 'top_middle', 'top_right',
                        'middle_left', 'middle_middle', 'middle_right',
                        'bot_left', 'bot_middle', 'bot_right', 'result')

X = tictactoe[1:9]
y = tictactoe[10]


set.seed(42)
#get train indexes - stratified samples
train_ix = createDataPartition(tictactoe$result, p = 0.85, list = FALSE)

#get train, test sets based on index
X_train = tictactoe[train_ix, ]
X_test = tictactoe[-train_ix, ]

set.seed(42)
#set train.control
tc = tune.control(sampling = "fix", fix = 9/10)

train.control = trainControl(method = 'cv', number = 10)

#FIRST METHOD USING CTREE
#use basic ctree without tuning any parameters
#Try to use ctree without cross-validation
X_train1 = lapply(X_train, factor)
X_test1  = lapply(X_test,  factor)

mCtree = ctree(result~., data = X_train1)
mCtree_predictions = predict(mCtree, newdata = X_train1[1:9])
cat("Accuracy of tree model on train set = ", mean(mCtree_predictions == X_train1$result))
cat("Accuracy of tree model on test set = ", mean(predict(mCtree, newdata = X_test1[1:9]) == X_test1$result))


#As expected, without k-fold cross validation, the accuracy went down

tree_grid = expand.grid(mincriterion = c(0.2,0.3,0.5,.35))
set.seed(76)
best_tree = train(result~., data = X_train1, method = 'ctree',
                  trControl = train.control, tuneGrid = tree_grid)

cat("Accuracy of tree model on train set = ",
    mean(predict(object = best_tree, newdata = X_train1[1:9]) == X_train1$result))


#now apply the tree_model on test set
predictions_using_tree = predict(object = best_tree, newdata = X_test1[1:9])

#display accuracy
cat("Accuracy of tree model on test set = ", mean(predictions_using_tree == X_test1$result))

best_tree_mean = mean(best_tree$resample$Accuracy)
cat("Average accuracy scores of ctree with parameters tuned and 10-fold cross validations: ", best_tree_mean)

#plot accuracies of 10 trees
plot(1:10, best_tree$resample$Accuracy, type = 'b', xlab = "Tree #", ylab = "Accuracy")

# END CTREE


#SECOND METHOD USING RANDOM FOREST
install.packages("randomForest")
library(randomForest)
#try base random forest with all untuned parameters
set.seed(67)
mRF = randomForest(result~., data = X_train1)
cat("Accuracy of Random Forest model on train set = ", mean(predict(mRF, newdata = X_train1[1:9]) == X_train1$result))
cat("Accuracy of Random Forest model on test set = ", mean(predict(mRF, newdata = X_test1[1:9]) == X_test1$result))


#tune the rf
#use best.tune() from e1071 because train() from caret only allows
#tuning mtry but not ntree

set.seed(76)
best_rf = best.tune(method = randomForest, train.x = result~., data = X_train1, 
                    ranges = list(mtry = c(1:9), 
                                  ntree = c(10,20,50,100,150,200,250,300)),
                    tunecontrol = tc)


cat("Accuracy of Random Forest model on train set = ", mean(predict(best_rf, newdata = X_train1[1:9]) == X_train1$result))

cat("Accuracy of random forest model on test set = ",
    mean(predict(best_rf, newdata = X_test1[1:9]) == X_test1$result))

rf_mean_OOBerr = mean(best_rf$err.rate[,1])
best_rf_mean = 1 - rf_mean_OOBerr 

cat("Mean accuracy scores of Random Forest model with tuned parameters and cross-validation: ", best_rf_mean)
#No need to use cross-validation for random forest since it is already done internally


#THIRD METHOD: SUPPORT VECTOR MACHINE
#convert nominal data into numeric (using one hot encode)
#use dummyVars from caret package

library(e1071) #for svm
#train set binarized
X_svm = dummyVars("~.", data = X_train1[1:9])
onehotE_ttt = data.frame(predict(X_svm, newdata = X_train1[1:9]))
onehotE_ttt$result = as.factor(ifelse(X_train1$result == 'positive',0,1))

#test set binarized
X_svm2 = dummyVars("~.", data = X_test1[1:9])
onehotE_ttt2 = data.frame(predict(X_svm2, newdata = X_test1[1:9]))
onehotE_ttt2$result = as.factor(ifelse(X_test1$result == 'positive',0,1))

tc = tune.control(sampling = "fix", fix = 9/10)

#run SVM without tuning
mSVM = svm(result~., data = onehotE_ttt)
cat("Accuracy of SVM model on train set = ", mean(predict(mSVM, newdata = onehotE_ttt[1:27]) == onehotE_ttt$result))

cat("Accuracy of SVM model on test set = ",
    mean(predict(mSVM, newdata = onehotE_ttt2[1:27]) == onehotE_ttt2$result))

set.seed(76)
best_svm = best.tune(method = svm, train.x = result ~ ., 
          data = onehotE_ttt, 
          ranges = list(kernel = c("sigmoid", "polynomial", 'radial')), 
          tunecontrol = tc)

set.seed(76)
best_svm = train(result~., data = onehotE_ttt, method = 'svmPoly',
                   trControl = train.control)
cat("Accuracy of SVM model on train set = ",
  mean(predict(svm_model, newdata = onehotE_ttt[1:27]) == onehotE_ttt$result))
cat("Accuracy of SVM model on train set = ",
    mean(predict(svm_model, newdata = onehotE_ttt2[1:27]) == onehotE_ttt2$result))
best_svm_mean = mean(best_svm$resample$Accuracy)
cat("Average accuracy of SVM with tuned parameters = ", best_svm_mean)
cat("Standard deviation of SVM model with tuned parameters = ", sd(best_svm$resample$Accuracy))

