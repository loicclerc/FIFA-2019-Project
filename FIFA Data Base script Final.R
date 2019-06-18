###################################Project FIFA 2019 Data Base#####################################

#The below code gives all the (relevant) analysis that was made in the frame of the project.
#In order to ensure that the running time is acceptable, only the code the gives the result
#for the most relevant parameters is set up to be run. The tunning part that are much more time
#consuming are put in comment. Of course, you are free to uncomment these parts if you wish.
#As it is now, running the code takes about xxx minutes on my computer. Running the full version
#takes xxx min on my computer.


#Packages
  if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
  if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
  if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
  if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
  if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
  if(!require(Rborist)) install.packages("Rborist", repos = "http://cran.us.r-project.org")
  if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")
  if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")


#In order to have consistent results everytime, the seed is set to 1:
  set.seed(1)


#Loading data
  dat_raw <- read.csv("https://raw.githubusercontent.com/loicclerc/FIFA-2019-Project/master/data.csv", stringsAsFactors = FALSE)

  
#Data Wrangling & Creation of the Database
  #Converting columns containing caracters into numbers. It is necessary to the scaling variable into account 
  #to multiply by the right factor
    dat_tuned <- dat_raw %>% mutate(Last_Char_Value = substr(dat_raw$Value, nchar(dat_raw$Value), nchar(dat_raw$Value)), 
                                Factor_Value = case_when(Last_Char_Value=="M" ~ 10^6,
                                                         Last_Char_Value=="K" ~ 10^3,
                                                         TRUE ~ 0),
                                Value = as.integer(case_when(Last_Char_Value==0 ~ 0,
                                                             TRUE ~ as.numeric(substr(dat_raw$Value, 4, nchar(dat_raw$Value)-1)) * 
                                                               Factor_Value)),
                                Last_Char_Wage = substr(dat_raw$Wage, nchar(dat_raw$Wage), nchar(dat_raw$Wage)), 
                                Factor_Wage = case_when(Last_Char_Wage=="K" ~ 10^3,
                                                        TRUE ~ 0),
                                Wage = as.integer(case_when(Last_Char_Wage==0 ~ 0,
                                                            TRUE ~ as.numeric(substr(dat_raw$Wage, 4, nchar(dat_raw$Wage)-1)) * 
                                                              Factor_Wage)),
                                Last_Char_Release.Clause = substr(dat_raw$Release.Clause, nchar(dat_raw$Release.Clause), 
                                                                  nchar(dat_raw$Release.Clause)),
                                Factor_Release.Clause = case_when(Last_Char_Release.Clause=="M" ~ 10^6,
                                                                  Last_Char_Release.Clause=="K" ~ 10^3,
                                                                  TRUE ~ 0),
                                Release.Clause = as.integer(case_when(Last_Char_Release.Clause=="" ~ 0,
                                                                      TRUE ~ as.numeric(substr(dat_raw$Release.Clause, 4, 
                                                                                               nchar(dat_raw$Release.Clause)-1)) * 
                                                                        Factor_Release.Clause))) %>%
  #Selection of the predictors that will be used to train the Database
    select(Age, Overall, Potential, Value, Wage, International.Reputation, 
           Weak.Foot, Skill.Moves, Crossing, Finishing, 
           HeadingAccuracy, ShortPassing, Volleys, Dribbling, Curve, FKAccuracy, 
           LongPassing, BallControl, Acceleration, SprintSpeed, Agility, Reactions, 
           Balance, ShotPower, Jumping, Stamina, Strength, LongShots, Aggression,
           Interceptions, Positioning, Vision, Penalties, Composure, Marking, StandingTackle, 
           SlidingTackle, GKDiving, GKHandling, GKKicking, GKPositioning, GKReflexes, Release.Clause)

  #Renaming problematic column names
    names(dat_tuned)[names(dat_tuned) == "International.Reputation"] <- "International_Reputation"
    names(dat_tuned)[names(dat_tuned) == "Skill.Moves"] <- "Skill_Moves"
    names(dat_tuned)[names(dat_tuned) == "Release.Clause"] <- "Release_Clause"

  #Replacing the na's by 0 (seems more meanignful as column average here)
    dat_tuned[is.na(dat_tuned)] <- 0

    
#Pre-processing: rescaling vectors
    
  #Separating predictors and outcome
    y <- dat_tuned$Value
    x <- dat_tuned %>% select(-Value)
    
  #Defining training and test set
    test_index <- createDataPartition(y=dat_tuned$Value, times = 1,
                                  p=0.2, list = FALSE)
    train_set_x <- x[-test_index,]
    train_set_y <- y[-test_index]
    test_set_x <- x[test_index,]
    test_set_y <- y[test_index]
    
  #Scaling the predictors
    preProc_x <- preProcess(train_set_x, method = "scale")
    train_set_x_scaled <- predict(preProc_x, train_set_x)
    test_set_x_scaled <- predict(preProc_x, test_set_x)
    
  #Rebuilding the dataframes
    train_set <-dat_tuned[-test_index,]
    test_set <-dat_tuned[test_index,]
    train_set_scaled <- train_set_x_scaled %>% mutate(Value = train_set_y)
    test_set_scaled <- test_set_x_scaled %>% mutate(Value = test_set_y)
  
  #removing intermediate variables
    rm(x, y, train_set_x, test_set_x, test_set_x_scaled, test_set_y, preProc_x)

    
#Functions to calculate the errors
  #function RMSE
    RMSE <- function(true_values, predicted_values){
      sqrt(mean((true_values - predicted_values)^2))
    }
    
  #function WMAPE
    WMAPE <- function(true_values, predicted_values){
      sum(abs(true_values - predicted_values))/sum(true_values)
    }


#Modeling --- Blocs code that are less relevant or more time consuming are put in comment
  #Global Average
    #Computating the global average to have a comparison basis
    Global_Average <- mean(train_set_scaled$Value)
    RMSE_Global_Average <- RMSE(test_set_scaled$Value, Global_Average)
    WMAPE_Global_Average <- WMAPE(test_set_scaled$Value, Global_Average)
    
  #Table to compare the results of all different approaches
    rmse_results <- data_frame(method = "Global Average" , RMSE = RMSE_Global_Average, WMAPE = WMAPE_Global_Average)
  
  #Lm model
    fit_lm <- lm(Value~., data =train_set_scaled)
    y_hat_Value <- predict(fit_lm, test_set_scaled)
    RMSE_lm <- RMSE(test_set_scaled$Value, y_hat_Value)
    WMAPE_lm <- WMAPE(test_set_scaled$Value, y_hat_Value)
    rmse_results <- bind_rows(rmse_results, data_frame(method="Lm model",
                                                     RMSE = RMSE_lm, WMAPE=WMAPE_lm))
                                                   
  #glm model
    train_glm <- train(Value~., method = "glm", data = train_set_scaled)
    y_hat_Value <- predict(train_glm, test_set_scaled, type="raw")
    RMSE_glm <- RMSE(test_set_scaled$Value, y_hat_Value)
    WMAPE_glm <- WMAPE(test_set_scaled$Value, y_hat_Value)
    rmse_results <- bind_rows(rmse_results, data_frame(method="glm model",
                                                     RMSE = RMSE_glm, WMAPE=WMAPE_glm))
    
  # knn model - hidden as long and not delivering good results
    # train_knn <- train(Value~., method = "knn", data = train_set_scaled)
    # y_hat_Value <- predict(train_knn, test_set_scaled, type="raw")
    # RMSE_knn <- RMSE(test_set_scaled$Value, y_hat_Value)
    # WMAPE_knn <- WMAPE(test_set_scaled$Value, y_hat_Value)
    # rmse_results <- bind_rows(rmse_results, data_frame(method="knn model",
    #                                                    RMSE = RMSE_knn, WMAPE=WMAPE_knn))
  
  #knn model - optimizing k with cv and testing more k's - hidden as long and not delivering good results
    # control <- trainControl(method = "cv", number = 10, p=.9)
    # train_knn_k <- train(Value~., method="knn",
    #                      data = train_set_scaled,
    #                      tuneGrid = data.frame(k=seq(1,19,2)),
    #                      trControl = control)
    # y_hat_Value <- predict(train_knn_k, test_set_scaled, type="raw")
    # RMSE_knn_k <- RMSE(test_set_scaled$Value, y_hat_Value)
    # WMAPE_knn_k <- WMAPE(test_set_scaled$Value, y_hat_Value)
    # rmse_results <- bind_rows(rmse_results, data_frame(method="knn cv",
    #                                                   RMSE = RMSE_knn_k, WMAPE=WMAPE_knn_k))
  
  #Rpart model
    fit_rpart <- rpart(Value~., data=train_set_scaled)
    y_hat_Value <- predict(fit_rpart, test_set_scaled)
    RMSE_rpart <- RMSE(test_set_scaled$Value, y_hat_Value)
    WMAPE_rpart <- WMAPE(test_set_scaled$Value, y_hat_Value)
    rmse_results <- bind_rows(rmse_results, data_frame(method="Rpart model",
                                                       RMSE = RMSE_rpart, WMAPE=WMAPE_rpart))
  
  #Rpart model - optimizing cp with cv - hidden as long and not delivering good results
    # train_rpart <- train(Value~.,
    #                      method = "rpart",
    #                      tuneGrid = data.frame(cp=seq(0.001,0.05,len=50)),
    #                      data = train_set_scaled)  
    # ggplot(train_rpart)
    # y_hat_Value <- predict(train_rpart, test_set_scaled)
    # RMSE_rpart_cp <- RMSE(test_set_scaled$Value, y_hat_Value)
    # WMAPE_rpart_cp <- WMAPE(test_set_scaled$Value, y_hat_Value)
    # rmse_results <- bind_rows(rmse_results, data_frame(method="RMSE rpart cp",
    #                                                    RMSE = RMSE_rpart_cp, WMAPE=WMAPE_rpart_cp))
  
  #Random Forest model - hidden because a better tuning parameter are found below
    # fit_rf <- randomForest(Value~., data=train_set_scaled)
    # y_hat_Value <- predict(fit_rf, test_set_scaled)
    # RMSE_rf <- RMSE(test_set_scaled$Value, y_hat_Value)
    # WMAPE_rf <- WMAPE(test_set_scaled$Value, y_hat_Value)
    # rmse_results <- bind_rows(rmse_results, data_frame(method="RF model",
    #                                                    RMSE = RMSE_rf, WMAPE=WMAPE_rf))
    
  #xgbTree model - hidden because a better tuning parameter are found below
    # train_xgbTree <- train(Value~., method = "xgbTree", data = train_set_scaled)
    # y_hat_Value <- predict(train_xgbTree, test_set_scaled, type="raw")
    # RMSE_xgbTree <- RMSE(test_set_scaled$Value, y_hat_Value)
    # WMAPE_xgbTree <- WMAPE(test_set_scaled$Value, y_hat_Value)
    
  #gamboost model
    train_gamboost <- train(Value~., method = "gamboost", data = train_set_scaled)
    y_hat_Value <- predict(train_gamboost, test_set_scaled, type="raw")
    RMSE_gamboost <- RMSE(test_set_scaled$Value, y_hat_Value)
    WMAPE_gamboost <- WMAPE(test_set_scaled$Value, y_hat_Value)
    rmse_results <- bind_rows(rmse_results, data_frame(method="gamboost model",
                                                       RMSE = RMSE_gamboost, WMAPE=WMAPE_gamboost))
    
  #gamLoess model
    train_gamLoess <- train(Value~., method = "gamLoess", data = train_set_scaled)
    y_hat_Value <- predict(train_gamLoess, test_set_scaled, type="raw")
    RMSE_gamLoess <- RMSE(test_set_scaled$Value, y_hat_Value)
    WMAPE_gamLoess <- WMAPE(test_set_scaled$Value, y_hat_Value)
    rmse_results <- bind_rows(rmse_results, data_frame(method="gamLoess model",
                                                       RMSE = RMSE_gamLoess, WMAPE=WMAPE_gamLoess))
  
  
  #rborist model - hidden as long and not delivering good results
    # control <- trainControl(method = "cv", number = 5, p=0.8)
    # grid <- expand.grid(minNode=seq(1,5,2), predFixed=seq(22,42,10))
    # train_rborist <- train(train_set_x_scaled,
    #                   train_set_y,
    #                   method = "Rborist",
    #                   nTree=50,
    #                   trControl=control,
    #                   tuneGrid=grid,
    #                   nSamp=5000)
    # fit_rborist <- Rborist(train_set_x_scaled,
    #                   train_set_y,
    #                   nTree=1000,
    #                   minNode=train_rborist$bestTune$minNode,
    #                   predFixed = train_rborist$bestTune$predFixed)
    # y_hat_Value <- predict(train_rborist, test_set_scaled)
    # RMSE_rborist <- RMSE(test_set_scaled$Value, y_hat_Value)
    # WMAPE_rborist <- WMAPE(test_set_scaled$Value, y_hat_Value)
  
  #Random Forest model - Best tunning paramter
    fit_rf_mtry_20 <- randomForest(Value~., data=train_set_scaled, mtry=20)
    y_hat_Value_mtry_20 <- predict(fit_rf_mtry_20, test_set_scaled)
    RMSE_rf_mtry_20 <- RMSE(test_set_scaled$Value, y_hat_Value_mtry_20)
    WMAPE_rf_mtry_20 <- WMAPE(test_set_scaled$Value, y_hat_Value_mtry_20)
    rmse_results <- bind_rows(rmse_results, data_frame(method="RF best tune",
                                                       RMSE = RMSE_xgbTree, WMAPE=WMAPE_xgbTree)) 
  
  #xgbTree model - better training parameters - hidden as long
    # train_xgbTree <- train(Value~., method = "xgbTree", data = train_set_scaled,
    #                        tuneGrid=expand.grid(nrounds = c(150, 250, 800), max_depth =3, eta=c(0.3, 0.4), 
    #                                             gamma=0, colsample_bytree = 0.8, min_child_weight=1, subsample=1), 
    #                        trcontrol = trainControl(method = "cv", number = 5, p=0.8))
    # y_hat_Value <- predict(train_xgbTree, test_set_scaled, type="raw")
    # RMSE_xgbTree <- RMSE(test_set_scaled$Value, y_hat_Value)
    # WMAPE_xgbTree <- WMAPE(test_set_scaled$Value, y_hat_Value)
    
  #xgbTree model - best tunning parameters
    train_xgbTree_Best <- train(Value~., method = "xgbTree", data = train_set_scaled,
                           tuneGrid=expand.grid(nrounds = 800, max_depth =3, eta=0.3, 
                                                gamma=0, colsample_bytree = 0.8, min_child_weight=1, subsample=1), 
                           trcontrol = trainControl(method = "cv", number = 5, p=0.8))
    y_hat_Value_Best_xgbTree <- predict(train_xgbTree, test_set_scaled, type="raw")
    RMSE_xgbTree_Best <- RMSE(test_set_scaled$Value, y_hat_Value_Best_xgbTree)
    WMAPE_xgbTree_Best <- WMAPE(test_set_scaled$Value, y_hat_Value_Best_xgbTree)
    rmse_results <- bind_rows(rmse_results, data_frame(method="xgbTree best tune",
                                                       RMSE = RMSE_xgbTree_Best, WMAPE=WMAPE_xgbTree_Best))
  
  #Result summary
  rmse_results %>% knitr::kable()  
