# Link to Data: 
# https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data?select=diabetes_binary_health_indicators_BRFSS2015.csv

# Steps:
# libraries
# import data
# make model
# check multicolinearity and assumptions
# graph it
# check correlations
# plot correlations and heat map
# check models
# validate models
# check accuracy of model
# check area under curve 

library(tidyverse)
library(ggplot2)
library(car)
library(dplyr)
library(corrplot)
library(pROC)

diabetes <- read.csv("diabetes binary.csv", header = TRUE)
head(diabetes)
nrow(diabetes)

# setting up model with every factor
model <- glm(Diabetes_binary ~ ., data = diabetes, family = binomial)
summary(model)

# checking multicollineaerity 
vif(model)
mean(vif(model))

# all our variables are < 5 or 10 so it's okay
# and because mean(vif(model)) is near 1, that means that multicollineartity 
# is not an issue
# with VIF we see multicollinearity is not an issue

#------------------------------------------------------------------------------#
# Assumption checking

# plot residuals, qq plot, scale location, and residual vs leverage 
par(mfrow = c(2,2))
plot(model) # plotting like this for analytics poster
# plot(model,which = 1) # Residual VS Fitted
# plot(model,which = 2) # QQ Residuals
# plot(model,which = 3) # Scale Location
# plot(model,which = 5) # Residual VS Leverage

#------------------------------------------------------------------------------#
# plotting correlations and heat map

#correlations looking at just diabetes_binary vs the factors
correlations <- cor(diabetes)[, "Diabetes_binary"]
correlations

# changing to data frame so I can plot it 
# need the factors (names) and the correlation values
cor_df <- data.frame(
  Factors = names(correlations),
  Correlation = correlations)

par(mfrow = c(1,1)) # resetting graphing back to normal 

# plotting - had to flip coords because of the labels being close to each other,
# it looked nasty 
ggplot(cor_df, aes(x = reorder(Factors, Correlation),
                   y = Correlation)) +
  geom_col(fill = "skyblue")+
  geom_text(aes(label = round(Correlation,3), hjust=0.0011))  +
  coord_flip()+
  theme_minimal() +
  labs(title = "Correlation with Diabetes",
       x = "Variable",
       y = "Correlation with Diabetes") + 
  theme(plot.title = element_text(hjust = 0.5))

# Plotting heat map
corrplot(cor(diabetes),method = "color",
         tl.col = "BLACK", tl.cex = 0.85,
         col=colorRampPalette(c("purple", "white", "red"))(10),
         addCoef.col = "black", number.cex = 0.49)

#------------------------------------------------------------------------------#
# Creating Stepwise + AIC process; Checking which step model is better
# Lecture Notes

# since vif is okay, I'm going to start stepwise process (AIC BIC stuff)
full_model <- glm(Diabetes_binary ~ ., data = diabetes)
null_model <- glm(Diabetes_binary ~ 1, data = diabetes)

forward_model <- step(null_model, scope = list(lower = null_model,
                                               upper = full_model),
                      direction = "forward")
summary(forward_model)

backward_model <- step(full_model, direction = "backward")
summary(backward_model)

stepwise_model <- step(null_model, scope = list(lower = null_model,
                                                upper = full_model),
                       direction = "both")
summary(stepwise_model)

#-----------------------------Compare the Models-------------------------------#
# AIC | BIC | adj R^2 | RSE 
# Lecture notes

# AIC values
aic_forward <- AIC(forward_model)
aic_backward <- AIC(backward_model)
aic_stepwise <- AIC(stepwise_model)

# BIC values
bic_forward <- BIC(forward_model)
bic_backward <- BIC(backward_model)
bic_stepwise <- BIC(stepwise_model)

# Adjusted Rˆ2 values
# https://www.geeksforgeeks.org/how-to-calculate-r-squared-for-glm-in-r/
forward_deviance <- summary(forward_model)$deviance
forward_null_deviance <- summary(forward_model)$null.deviance
adjr2_forward <- 1 - (forward_deviance / forward_null_deviance)

backward_deviance <- summary(backward_model)$deviance
backward_null_deviance <- summary(backward_model)$null.deviance
adjr2_backward <- 1 - (backward_deviance / backward_null_deviance)

stepwise_deviance <- summary(stepwise_model)$deviance
stepwise_null_deviance <- summary(stepwise_model)$null.deviance
adjr2_stepwise <- 1 - (stepwise_deviance / stepwise_null_deviance)

# RSE values
# https://www.statology.org/residual-standard-error-r/
rse_forward <- sqrt(deviance(forward_model)/df.residual(forward_model))
rse_backward <- sqrt(deviance(backward_model)/df.residual(backward_model))
rse_stepwise <- sqrt(deviance(stepwise_model)/df.residual(stepwise_model))

# stuff from lecture notes that didn't work, had to change since GLM
# see code for R2 and RSE above

#adjr2_forward <- summary(forward_model)$adj.r.squared
#adjr2_backward <- summary(backward_model)$adj.r.squared
#adjr2_stepwise <- summary(stepwise_model)$adj.r.squared

# Residual Standard Error (RSE)
#rse_forward <- summary(forward_model)$sigma
#rse_backward <- summary(backward_model)$sigma
#rse_stepwise <- summary(stepwise_model)$sigma

# Display the results
results <- data.frame(
  Method = c("Forward Selection", "Backward Elimination", "Stepwise Selection"),
  AIC = c(aic_forward, aic_backward, aic_stepwise),
  BIC = c(bic_forward, bic_backward, bic_stepwise),
  Adjusted_R2 = c(adjr2_forward, adjr2_backward, adjr2_stepwise),
  RSE = c(rse_forward, rse_backward, rse_stepwise))
print(results)

#------------------------------------------------------------------------------#

#finding odds ratio
ratios <- exp(coef(stepwise_model))
sort(ratios, decreasing = TRUE)

#------------------------------------------------------------------------------#
#Model Validation with RMSE, MAE, R^2
# Lecture Notes

# Load necessary libraries
library(caret)   # for train/test split and MAE/RMSE
#install.packages("Metrics")
library(Metrics) # for MAE, RMSE (optional)
set.seed(123)    # for reproducibility

# Create train/test split (80% train, 20% test)
trainIndex <- createDataPartition(diabetes$Diabetes_binary, p = 0.8,
                                  list = FALSE)
trainData <- diabetes[trainIndex, ]
testData  <- diabetes[-trainIndex, ]

#---------- Finding Optimal Train/Test Split ----------------

# Function to evaluate model performance for a given split
evaluate_model <- function(split_ratio) {
  rmse_vals <- c()
  for (i in 1:10) {  # Repeat 10 times to average out variation
    trainIndex <- createDataPartition(diabetes$Diabetes_binary, p = split_ratio,
                                      list = FALSE)
    trainData <- diabetes[trainIndex, ]
    testData  <- diabetes[-trainIndex, ]
    
    model <- glm(Diabetes_binary ~ ., data = trainData, family = binomial)
    preds <- predict(model, newdata = testData)
    rmse_vals[i] <- RMSE(preds, testData$Diabetes_binary)
  }
  return(mean(rmse_vals))
}

# Try different train/test splits
split_ratios <- seq(0.5, 0.9, by = 0.05)
rmse_results <- sapply(split_ratios, evaluate_model)

# Identify optimal split
optimal_split <- split_ratios[which.min(rmse_results)]
cat("Optimal Train/Test Split:", round(optimal_split * 100), "/",
    round((1 - optimal_split) * 100), "\n")

# Re-train model using optimal split
trainIndex <- createDataPartition(diabetes$Diabetes_binary, p = optimal_split,
                                  list = FALSE)
trainData <- diabetes[trainIndex, ]
testData  <- diabetes[-trainIndex, ]

#---------------------------------------------------------
# Lecture Notes Cont.

# Fit linear model on training data
model <- glm(Diabetes_binary ~ ., data = trainData, family = binomial)

# Predict on test data
predictions <- predict(model, newdata = testData, type = "response")

# Actual values
actuals <- testData$Diabetes_binary

# --- Model Performance ---
# 1. RMSE - Root Mean Square Error
rmse_val <- RMSE(predictions, actuals)
print(paste("RMSE:", round(rmse_val, 4)))
# Measures Average squared error
# More sensitive to outliers
# Lower is the better. 

# 2. MAE - Mean Absolute Error
mae_val <- MAE(predictions, actuals)  
print(paste("MAE:", round(mae_val, 4)))
# Measures Average absolute prediction error
# Less sensitive to outliers
# Lower is the better

# 3. R-squared (on test data)
sst <- sum((actuals - mean(actuals))^2)
sse <- sum((actuals - predictions)^2)
r_squared <- 1 - (sse / sst)
print(paste("R-squared:", round(r_squared, 4)))
# Measures proportion of variance in response variable explained by the model.
# < 0.3	Weak fit (not explaining much)
# 0.3 – 0.5	Moderate fit
# 0.5 – 0.7	Good fit
# 0.7 – 0.9	Strong fit
# > 0.9	Very strong fit 

performance = c(rmse_val, mae_val, r_squared)
performance
#------------------------------------------------------------------------------#
# Checking Accuracy

# https://www.linkedin.com/advice/0/how-do-you-measure-machine-learning-model-accuracy?utm_source=share&utm_medium=member_desktop&utm_campaign=copy
accuracy <- (actuals/predictions)
mean(accuracy)

#------------------------------------------------------------------------------#
# Area under curve
roc_curve <- roc(actuals, predictions)
value <- round(auc(actuals, predictions),2)
plot(roc_curve)
value
text(0.5, 0.2, paste("AUC = ", value))

