install.packages("imputeTS")
install.packages("NeuralNetTools")
install.packages("glm2")
install.packages("stats")
install.packages("corrplot")
install.packages("nnet")

library(NeuralNetTools)
library(nnet)
library(corrplot)
library(stats)
library(glm2)
library(imputeTS)
library(psych)
library(dplyr)
library(ggpubr)
library(ggplot2)
library(neuralnet)
library(plyr) 
library(caret)

campus <- read.csv("Campus_Data.csv",stringsAsFactors = FALSE)

str(campus)

campus <- data.frame(campus)
campus <- campus[6:22]
campus <- unlist(campus)  ##unlisting required to change numbers to numeric
campus <- as.numeric(campus)

campus <- matrix(campus, ncol = 17, nrow = 234, byrow = FALSE)
campus <- as.data.frame(campus)


headers <- c("D_Campus_Dropout","D_Male", "D_Female", "D_Black", 
           "D_Asian", "D_Hispanic", "D_Native_American", "D_White", "D_Pacific_Islander", 
           "D_Multiracial", "D_At_Risk", "D_Bilingual", "D_Econ_Disadv", "D_Homeless", 
           "D_Immigrant", "D_LEP", "D_SPED")

campus <- data.frame(campus)
names(campus) = headers

## Handle Missing Values: We will be taking the mean of each feature 
## and replacing all NAs of that column with that mean value.


missing_values <- colnames(campus)[apply(campus, 2, anyNA)]  ## Columns with missing values



mean_fun <- function(x) {
  mean(x, na.rm = TRUE)
}

means <- apply(campus,2,mean_fun)

campus$D_Native_American <-  na.replace(campus$D_Native_American,
                                        mean(campus$D_Native_American, na.rm = TRUE))

campus$D_Male <-  na.replace(campus$D_Male,
                                        mean(campus$D_Male, na.rm = TRUE))

campus$D_Female <-  na.replace(campus$D_Female,
                                        mean(campus$D_Female, na.rm = TRUE))

campus$D_Black <-  na.replace(campus$D_Black,
                                        mean(campus$D_Black, na.rm = TRUE))

campus$D_Asian <-  na.replace(campus$D_Asian,
                                        mean(campus$D_Asian, na.rm = TRUE))

campus$D_White <-  na.replace(campus$D_White,
                                        mean(campus$D_White, na.rm = TRUE))

campus$D_Pacific_Islander <-  na.replace(campus$D_Pacific_Islander,
                                        mean(campus$D_Pacific_Islander, na.rm = TRUE))

campus$D_Multiracial <-  na.replace(campus$D_Multiracial,
                                        mean(campus$D_Multiracial, na.rm = TRUE))

campus$D_At_Risk <-  na.replace(campus$D_At_Risk,
                                        mean(campus$D_At_Risk, na.rm = TRUE))

campus$D_Bilingual <-  na.replace(campus$D_Bilingual,
                                        mean(campus$D_Bilingual, na.rm = TRUE))

campus$D_Econ_Disadv <-  na.replace(campus$D_Econ_Disadv,
                                        mean(campus$D_Econ_Disadv, na.rm = TRUE))

campus$D_Homeless <-  na.replace(campus$D_Homeless,
                                        mean(campus$D_Homeless, na.rm = TRUE))

campus$D_Immigrant <-  na.replace(campus$D_Immigrant,
                                        mean(campus$D_Immigrant, na.rm = TRUE))

campus$D_LEP <-  na.replace(campus$D_LEP,
                                        mean(campus$D_LEP, na.rm = TRUE))

campus$D_SPED <-  na.replace(campus$D_SPED,
                                        mean(campus$D_SPED, na.rm = TRUE))


# Check correlation between all features
cor_features <- cor(campus)

View(cor_features)

## The Male and Female features have a high correlation with Campus Dropout Rate, as well 
## as with other features. Remove them both.

campus <- campus[-3]
campus <- campus[-2]


## Sampling: We will be doing a 75:25 split with randomization

set.seed(123)

p <- sample(234, 175)

campus_train <- campus[p, ]

campus_test <- campus[-p, ]


# Fitting a linear model and checking  #

lm_fit <- glm(D_Campus_Dropout ~., data = campus_train)

summary(lm_fit)



# Using lm_fit to make predictions on test data #

pr_lm <- predict(lm_fit, campus_test)


# Test MSE #

MSE_lm <- sum((pr_lm - campus_test$D_Campus_Dropout)^2)/nrow(campus_test)


# Scaling data for the ANN Algorithm to run properly since ANN requires data 
# to normalized and scaled.

lin <- function(x){

return(log(x+1))
}

norm <- function(x){

return((x - min(x))/(max(x) - min(x)))
}


# Applying the lin and norm functions to the data and creating randomized train 
# and test datasets for Ann #

campus_n <- as.data.frame(lapply(campus, norm))

campus_n$D_Campus_Dropout <- sapply(campus_n$D_Campus_Dropout, lin)

set.seed(123)

d <- sample(234, 175)

campus_train_n <- campus_n[d, ]

campus_test_n  <- campus_n[-d, ]


# ANN training: Train the data using the model ANN 
# with two hidden layers (one being 5 nodes and the second being 3 nodes).
# Linear Output will be true.

n  <- names(campus_train_n)

f  <- as.formula(paste("D_Campus_Dropout ~", paste(n[!n %in% "D_Campus_Dropout"], collapse = " + ")))

nn <- neuralnet(f, data=campus_train_n, hidden = c(5, 3), linear.output = T)


# Visual plot of the model #

plot(nn)


# Using ANN model to get predictions #

pr_nn <- compute(nn, campus_test_n[ , 2:15])


# Results from NN are normalized (scaled) #

# Descaling for comparison #

pr.nn_ <- pr_nn$net.result*(max(campus$D_Campus_Dropout ) - min(campus$D_Campus_Dropout )) + min(campus$D_Campus_Dropout )

test.r <- (campus_test_n$D_Campus_Dropout )*(max(campus$D_Campus_Dropout )- min(campus$D_Campus_Dropout )) + min(campus$D_Campus_Dropout )


# Calculating MSE #

MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(campus_test_n)


# Compare the two MSEs #

print(paste(MSE_lm, MSE.nn))


# Plot predictions #

par(mfrow=c(1, 2))

plot(campus_test$D_Campus_Dropout, pr.nn_, col='red', main='Real vs predicted NN', pch=18, cex = 0.7)

abline(0, 1, lwd = 2)

legend('bottomright', legend = 'NN', pch = 18, col = 'red', bty = 'n')

plot(campus_test$D_Campus_Dropout, pr_lm, col = 'blue', main = 'Real vs predicted lm', pch = 18, cex = 0.7)

abline(0,1,lwd=2)

legend('bottomright', legend = 'LM', pch = 18, col = 'blue', bty = 'n', cex = .95)


# Compare predictions on the same plot #

plot(campus_test$D_Campus_Dropout, pr.nn_, col = 'red', main = 'Real vs predicted NN', pch = 18, cex = 0.7)

points(campus_test$D_Campus_Dropout, pr_lm, col = 'blue', pch = 18, cex = 0.7)

abline(0, 1, lwd = 2)

legend('bottomright', legend = c('NN','LM'), pch = 18, col = c('red', 'blue'))


# Cross validating 

library(boot)

set.seed(123)


# Linear model cross validation # RUQIA

lm.fit <- glm(D_Campus_Dropout~., data = campus_train_n)

cv.glm(campus_train_n, lm.fit, K = 10)$delta[1]


# Neural net cross validation #

set.seed(123)

cv.error <- NULL

k <- 10


# Initialize progress bar #

pbar <- create_progress_bar('text')

pbar$init(k)



for(i in 1:k){

 
    train.cv <- campus_train_n

    test.cv  <-  campus_test_n

    

    nn <- neuralnet(f, data = train.cv, hidden = c(5, 2), linear.output = T)

    

    pr.nn <- compute(nn, test.cv[, 2:15])

    pr.nn <- pr.nn$net.result*(max(campus$D_Campus_Dropout) - min(campus$D_Campus_Dropout)) + min(campus$D_Campus_Dropout)

    

    test.cv.r <- (test.cv$D_Campus_Dropout)*(max(campus$D_Campus_Dropout) - min(campus$D_Campus_Dropout)) + min(campus$D_Campus_Dropout)

    

    cv.error[i] <- sum((test.cv.r - pr.nn)^2)/nrow(test.cv)

    

    pbar$step()

}



# Average MSE #

mean(cv.error)


# MSE vector from CV #

cv.error


# Visual plot of CV results #

boxplot(cv.error, xlab = 'MSE CV', col = 'cyan',

        border = 'blue', names = 'CV error (MSE)',

        main = 'CV error (MSE) for NN', horizontal = TRUE)


# Displaying the importance of each feature #

olden(nn)



# Correaltion Plot: Let's see which features have a strong correlation to each other

corrplot(cor(campus_n))

corrplot(cor(campus_train_n))

corrplot(cor(campus_test_n))

# Dispalying the sensitivity of each feature


lekprofile(nn)


##

ctrl <- trainControl(method = "CV", number = 10)


grid <- expand.grid(.size = 6, .decay = .1:.9)


fitControl <- train(D_Campus_Dropout~ ., data = campus_train_n, method = "nnet", metric = "Rsquared", trControl = ctrl, tundeGrid = grid)
















