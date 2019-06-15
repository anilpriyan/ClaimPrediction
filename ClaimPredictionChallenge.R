
# author: "Anil Nanayakkara"

  
############################################################################################################
#
# packages

  if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
  if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
  if(!require(gtools)) install.packages("gtools", repos = "http://cran.us.r-project.org")
  if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
  if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
  
  my_image <- function(x, zlim = range(x), ...){
    colors = rev(RColorBrewer::brewer.pal(9, "RdBu"))
    cols <- 1:ncol(x)
    rows <- 1:nrow(x)
    image(cols, rows, t(x[rev(rows),,drop=FALSE]), xaxt = "n", yaxt = "n",
          xlab="", ylab="",  col = colors, zlim = zlim, ...)
    abline(h=rows + 0.5, v = cols + 0.5)
    axis(side = 1, cols, colnames(x), lwd = 0.5, las = 2)
    axis(side = 2, rows, rev(rownames(x)), lwd = 0.5, las = 1, cex.axis = 1)
  }
#
#
#############################################################################################################



##################################################################################################################
#
# setup the data
  unzip("train_set.zip")
  train_set <- read.csv("train_set.csv")
  train_set_amt_cat <- train_set %>% mutate(Amount_Category = as.factor(case_when(
    Claim_Amount > 0.00 ~ "CLAIM",
    Claim_Amount == 0.00 ~ "NO_CLAIM"
  )))
  rm(train_set)
  total_row_count <- nrow(train_set_amt_cat)
  sequences <- seq(1, total_row_count, by = 500000)
  claim_data_2005 <- data.frame()
  claim_data_2006 <- data.frame()
  claim_data_2007 <- data.frame()
  Reduce(function(x, y) {
    filtered_by_2005 <- train_set_amt_cat[x:y-1,] %>% filter(Calendar_Year == 2005)
    claim_data_2005 <<- rbind(claim_data_2005, filtered_by_2005)
    filtered_by_2006 <- train_set_amt_cat[x:y-1,] %>% filter(Calendar_Year == 2006)
    claim_data_2006 <<- rbind(claim_data_2006, filtered_by_2006)
    filtered_by_2007 <- train_set_amt_cat[x:y-1,] %>% filter(Calendar_Year == 2007)
    claim_data_2007 <<- rbind(claim_data_2007, filtered_by_2007)
    y
  }, sequences)
  claim_data_2005 <- rbind(claim_data_2005, train_set_amt_cat[max(sequences):total_row_count,] %>%
                             filter(Calendar_Year == 2005))
  claim_data_2006 <- rbind(claim_data_2006, train_set_amt_cat[max(sequences):total_row_count,] %>%
                             filter(Calendar_Year == 2006))
  claim_data_2007 <- rbind(claim_data_2007, train_set_amt_cat[max(sequences):total_row_count,] %>%
                             filter(Calendar_Year == 2007))
  rm(train_set_amt_cat)
  
  glimpse(claim_data_2005)
#
#
############################################################################################################




################################################################################################################
#
# insured vehicle and claim numbers by year

    number_of_vehicles_insured_for_2005 <- nrow(claim_data_2005)
    number_of_vehicles_insured_for_2006 <- nrow(claim_data_2006)
    number_of_vehicles_insured_for_2007 <- nrow(claim_data_2007)
    number_of_total_records <- number_of_vehicles_insured_for_2005 + number_of_vehicles_insured_for_2005 + 
    number_of_vehicles_insured_for_2005
    number_of_claims_for_2005 <- sum(claim_data_2005$Amount_Category == "CLAIM")
    number_of_claims_for_2006 <- sum(claim_data_2006$Amount_Category == "CLAIM")
    number_of_claims_for_2007 <- sum(claim_data_2007$Amount_Category == "CLAIM")
    number_of_claims_for_2005_to_2007 <- number_of_claims_for_2005 + number_of_claims_for_2006 + number_of_claims_for_2007
    numbers_by_years <- data.frame()
    numbers_by_years <- rbind(numbers_by_years, data.frame(Year = "2005", 
    Vehicles  = number_of_vehicles_insured_for_2005, 
    Claims = number_of_claims_for_2005))
    numbers_by_years <- rbind(numbers_by_years, data.frame(Year = "2006", 
    Vehicles = number_of_vehicles_insured_for_2006, 
    Claims = number_of_claims_for_2006))
    numbers_by_years <- rbind(numbers_by_years, data.frame(Year = "2007", 
    Vehicles = number_of_vehicles_insured_for_2007, 
    Claims = number_of_claims_for_2007))
    numbers_by_years <- rbind(numbers_by_years, data.frame(Year = "ALL", 
    Vehicles = number_of_total_records, 
    Claims = number_of_claims_for_2005_to_2007))
    numbers_by_years %>% knitr::kable()
#
#
###############################################################################################################

###############################################################################################################
#
#   Observations
    
    #Appendix A
    
    small_set <- claim_data_2005[1:10000,]
    
    plot_continuous_var_combinations <- function(x, y) {
      small_set %>% ggplot(aes(x = .[[x]], y = .[,y], size = Claim_Amount)) +
        scale_size_continuous(range = c(1, 10)) +
        geom_point(aes(color = Claim_Amount > 0)) + labs(x = x, y = y)
    }
    
    continuous_vars <- c("Var1", "Var2", "Var3", "Var4", "Var5", "Var6", "Var7", "Var8")
    
    continuous_var_combos <- combinations(8, 2, continuous_vars)
    
    # display only four combinations for brevity, but can plot 28 combinations
    lapply(1:4, function(i) {
      print(plot_continuous_var_combinations(continuous_var_combos[i,1], continuous_var_combos[i,2]))
    })
    
    
    
    # Appendix B
    
    logistic <- glm(Amount_Category ~ Cat1, data = claim_data_2005[1:10000,], family = "binomial")
    
    summary(logistic)
    
    logistic <- glm(Amount_Category ~ Cat2, data = claim_data_2005[1:10000,], family = "binomial")
    
    summary(logistic)
    
    logistic <- glm(Amount_Category ~ Cat4, data = claim_data_2005[1:10000,], family = "binomial")
    
    summary(logistic)
    
    logistic <- glm(Amount_Category ~ Cat5, data = claim_data_2005[1:10000,], family = "binomial")
    
    summary(logistic)
    
    logistic <- glm(Amount_Category ~ Cat6, data = claim_data_2005[1:10000,], family = "binomial")
    
    summary(logistic)
    
    logistic <- glm(Amount_Category ~ Cat7, data = claim_data_2005[1:10000,], family = "binomial")
    
    summary(logistic)
    
    logistic <- glm(Amount_Category ~ Cat8, data = claim_data_2005[1:10000,], family = "binomial")
    
    summary(logistic)
    
    logistic <- glm(Amount_Category ~ Cat9, data = claim_data_2005[1:10000,], family = "binomial")
    
    summary(logistic)
    
    logistic <- glm(Amount_Category ~ OrdCat, data = claim_data_2005[1:10000,], family = "binomial")
    
    summary(logistic)
#
#
##############################################################################################################

################################################################################################################
#
# common functions

    accuracy_data <- function(method, byClassData) {
    data_frame(Method = method, 
    Sensitivity = byClassData['Sensitivity'],
    Specificity = byClassData['Specificity'],
    Balanced_Accuracy = byClassData['Balanced Accuracy'],
    F1 = byClassData['F1'])
    } 
    
    accuracy_data_results <- function(method, data_set) {
    conf_matrix <- 
    confusionMatrix(data = data_set$predicted, reference = data_set$Amount_Category)
    accuracy_classes <- conf_matrix$byClass
    accuracy <- accuracy_data(method, accuracy_classes)
    accuracy_results <- rbind(accuracy_results, accuracy)   
    }
    
    claim_data_with_claims_2005 <- claim_data_2005 %>% 
    filter(Amount_Category == "CLAIM")
    claim_data_with_no_claims_2005 <- claim_data_2005 %>% 
    filter(Amount_Category == "NO_CLAIM")
    claim_data_with_claims_2006 <- claim_data_2006 %>% 
    filter(Amount_Category == "CLAIM")
    claim_data_with_no_claims_2006 <- claim_data_2006 %>% 
    filter(Amount_Category == "NO_CLAIM")
    claim_data_with_claims_2007 <- claim_data_2007 %>% 
    filter(Amount_Category == "CLAIM")
    claim_data_with_no_claims_2007 <- claim_data_2007 %>% 
    filter(Amount_Category == "NO_CLAIM")
    
    overall_rate <- round(count(claim_data_with_claims_2005)/count(claim_data_2005)*10000, digits = 0)
    data_frame("Overall Claim Frequency per 10000" = overall_rate$n) %>% knitr::kable()
    
    accuracy_results <- data.frame()
#
#
################################################################################################################


################################################################################################################
#
# Counts of combinations specific categorical classes and numerical characteristics where claims occur

    cat_types <- c("Cat5_B", "Cat7_A", "Cat7_B", "Cat2_A", "Cat4_A", "Cat5_A", "Cat7_C", "Cat9_B")
    
    var_types <- c("Var1", "Var2", "Var3", "Var5", "Var6", "Var7", "Var8")
    
    var_strata <- c(-3.00, 0, 3.00)
    
    get_claim_count_by_cat_type_and_var_strata <- function(c, t) {
      # match with Cat types
      cats <- mapply(FUN = function(i) {
        split <- strsplit(i, "_")[[1]]
        cat <- split[1]
        cat_value <- split[2]
        cat_data_with_claims <- c[c[[cat]] == cat_value,]
        cat_data_total <- t[t[[cat]] == cat_value,]
        # count claim vehicle counts for Var strata - Cat type combination
        vars <- mapply(function(j) {
          mapply(FUN = function(k) {
            indexes_stratified_by_k_for_data_with_claims <- mapply(FUN = function(l) {
            # set true if value is closest to the strata.
            var_strata[which.min(abs(l - var_strata))] == k
            }, cat_data_with_claims[[j]])
            indexes_stratified_by_k_for_total_data <- mapply(FUN = function(l) {
            # set true if value is closest to the strata.
            var_strata[which.min(abs(l - var_strata))] == k
            }, cat_data_total[[j]])
            var_stratified_data_with_claims <-
              cat_data_with_claims[indexes_stratified_by_k_for_data_with_claims,]
            var_stratified_data_total <-
              cat_data_total[indexes_stratified_by_k_for_total_data,]
            rate <- round(unlist(count(var_stratified_data_with_claims))/
              unlist(count(var_stratified_data_total)) * 10000, digits = 0)
            rate_list <- ifelse(rate == 0, NA,  rate - overall_rate)
            unlist(rate_list)
          }, var_strata)
        }, var_types)
      }, cat_types)
    }
    
    claim_count_cat_type_var_strata_data_2005 <-
    get_claim_count_by_cat_type_and_var_strata(claim_data_with_claims_2005, claim_data_2005)
    
    rownames(claim_count_cat_type_var_strata_data_2005)[1:21] <- as.vector(sapply(var_types, function(i) {
    sapply(var_strata, function(j) {
    paste0(i, "_", j)
    })
    }))
    
    claim_count_cat_type_var_strata_data_2005 %>% knitr::kable()
#
#
################################################################################################################


################################################################################################################
#
# claim count percentages for categorical and numerical vehicle characteristics

  claim_count_cat_type_var_strata_data_2005 <- 
    claim_count_cat_type_var_strata_data_2005[order(rowMeans
    (claim_count_cat_type_var_strata_data_2005, na.rm = TRUE), decreasing = TRUE),
    order(colMeans(claim_count_cat_type_var_strata_data_2005,na.rm = TRUE))]
    claim_count_cat_type_var_strata_data_2005[is.na(claim_count_cat_type_var_strata_data_2005)] <- 0
  my_image(claim_count_cat_type_var_strata_data_2005)
#
#
#################################################################################################################


##################################################################################################################
#
# pca analysis
  
    set.seed(755)
    test_index <- createDataPartition(y = claim_data_2006$Amount_Category, times = 1,
                                      p = 0.05, list = FALSE)
    train_set_2006 <- claim_data_2006[-test_index,]
    test_set_2006 <- claim_data_2006[test_index,]

    get_cat_var_reads <- function(data_set) {
      mapply(FUN = function(x){
        mapply(FUN = function(i) {
          split <- strsplit(i, "_")[[1]]
          cat <- split[1]
          cat_value <- split[2]
          data_row <- data_set %>% filter(Row_ID == x)
          matched_interested_category <- data_row[[cat]] == cat_value
          
          vars <- mapply(function(j) {
            mapply(FUN = function(k) {
              stratified_by_k <- mapply(FUN = function(l) {
                matched_strata <- var_strata[which.min(abs(l - var_strata))] == k
                ifelse(matched_interested_category & matched_strata,
                       claim_count_cat_type_var_strata_data_2005[paste0(j, "_", k) , i],
                       0)
              }, data_row[[j]])
            }, var_strata)
          }, var_types, USE.NAMES = TRUE)
        }, cat_types)
      }, data_set$Row_ID)
    }
    
    generate_primary_component_data <- function(data_set) {
      samples_matrix <- get_cat_var_reads(data_set)
      PC1 <- apply(t(samples_matrix), 1, function(row) {
        crossprod(row, pca$rotation[,1]) * pca$x[1]
      })
      PC2 <- apply(t(samples_matrix), 1, function(row) {
        crossprod(row, pca$rotation[,2]) * pca$x[2]
      })
      Category <- data_set$Amount_Category
      Row_ID <- data_set$Row_ID
      pca.data_1_2 <- data.frame(Category, PC1, PC2, Row_ID)
    }
    
    samples_matrix_claim <- get_cat_var_reads(claim_data_with_claims_2005[1:2000,])
    colnames(samples_matrix_claim) <- rep("CLAIM", 2000)
    samples_matrix_no_claim <- get_cat_var_reads(claim_data_with_no_claims_2005[1:2000,])
    colnames(samples_matrix_no_claim) <- rep("NO_CLAIM", 2000)
    
    samples_matrix <- cbind(samples_matrix_claim, samples_matrix_no_claim)
    
    samples_matrix <- sweep(samples_matrix, 1,
                            rowMeans(samples_matrix, na.rm = TRUE))
    samples_matrix <- sweep(samples_matrix, 2,
                            colMeans(samples_matrix, na.rm = TRUE))
    
    samples_matrix[is.na(samples_matrix)] <- 0
    
    pca <- prcomp(t(samples_matrix))
    
    pca.var <- pca$sdev^2
    
    pca.var.per <- round(pca.var/sum(pca.var)*100,1)
    
    X1 <- apply(t(samples_matrix), 1, function(row) {
      crossprod(row, pca$rotation[,1]) * pca$x[1]
    })
    
    Y1 <- apply(t(samples_matrix), 1, function(row) {
      crossprod(row, pca$rotation[,2]) * pca$x[2]
    })
    
    Category <- rownames(t(samples_matrix))
    
    pca.data_1_2 <- data.frame(Category, X1, Y1)
    
    pca.data_1_2 %>% ggplot(aes(x = X1, y = Y1, label = Category)) +
      geom_point(aes(col = Category)) +
      geom_hline(yintercept = -105, linetype = "dotted") + 
      geom_vline(xintercept = -75, linetype = "dotted") +
      xlab(paste("PC1 - ", pca.var.per[1], "%", sep = "")) +
      ylab(paste("PC2 - ", pca.var.per[2], "%", sep = "")) +
      theme_bw() +
      scale_color_discrete(name = "Claim Class") +
      ggtitle("Claim Classification Cluster Analysis")
#
#
#####################################################################################################################


#################################################################################################################
#
#   pca prediction

    pca_prediction <- function(x, y, predict_data_set, primary_component_data) {
    # PC1 == 0 and PC2 == 0 implies that all the factors were zero.
    select_data <- primary_component_data %>% filter(PC1 > x & PC2 > y & PC1 != 0 & PC2 != 0)
    predicted_as_claim_data <- predict_data_set %>% semi_join(select_data, by = "Row_ID") %>%
    mutate(predicted = "CLAIM")
    predicted_as_no_claim_data <- predict_data_set %>% anti_join(select_data, by = "Row_ID") %>%
    mutate(predicted = "NO_CLAIM")
    predicted_data <- rbind(predicted_as_claim_data, predicted_as_no_claim_data)
    predicted_data$predicted <- factor(predicted_data$predicted, levels = c("CLAIM", "NO_CLAIM")) 
    predicted_data
  }
  
    input_data_set <- train_set_2006[1:10000,]
    pca_data <- generate_primary_component_data(input_data_set)
    cutoffs <- c(-75, -80, -85, -90, -95, -100, -105, -110, -115, -125)
    cutoff_combinations <- combinations(10, 2, cutoffs)
    
    pca_results <- map_df(1:45, function(i) {
    predicted_data <- pca_prediction(cutoff_combinations[i,1], cutoff_combinations[i,2], 
    input_data_set, pca_data)
    pca_accuracy_data <- accuracy_data_results(paste0("PC1 > ", cutoff_combinations[i,1], " , ", 
    "PC2 > ", cutoff_combinations[i,2]), predicted_data)
    })
    
    pca_results %>% knitr::kable()
#
#
##############################################################################################################


##################################################################################################################
#
#   logistic prediction
    
    logisic_regression_prediction <- function(cutoff, logistic, data_set) {
    predicted_probabilities <- predict(logistic, data_set, type = "response")
    data_set$predicted <- ifelse(predicted_probabilities > cutoff, "NO_CLAIM", "CLAIM")
    data_set$predicted <- factor(data_set$predicted, levels = c("CLAIM", "NO_CLAIM"))
    data_set
    }
    
    cutoff <- seq(0.98, 1, 0.001)
    
    logistic <- glm(Amount_Category ~ Cat1 + Cat2 + Cat5 + Cat7 + Cat9 + Var1 + Var2 +
    Var3 + Var7 + Var8,
    data = input_data_set, family = "binomial")
    
    logistic_results <- map_df(cutoff, function(x) {
    logistic_set <- logisic_regression_prediction(x, logistic, test_set_2006)
    logistic_accuracy_data <- accuracy_data_results(paste0("LOGISTIC REGRESSION - cutoff -> ", x), logistic_set)
    })
    
    
    logistic_results %>% knitr::kable()
#
#
###############################################################################################################


###############################################################################################################
#
# Validation against the test set with the Best results with PCA and logistic regression
    
    input_data_set <- test_set_2006[1:10000,]
    pca_best_pc1 <- -90
    pca_best_pc2 <- -75
    logistic_best_cutoff <- 0.996
    pca_data <- generate_primary_component_data(input_data_set)
    best_pca_result <- pca_prediction(pca_best_pc1, pca_best_pc2, input_data_set, pca_data)
    best_pca_accuracy_data <- accuracy_data_results(paste0("Test set PCA result: PC1 > ", pca_best_pc1, " , ", "PC2 > ", pca_best_pc2), 
    best_pca_result)
    
    best_pca_accuracy_data %>% knitr::kable()
    
    best_logistic_result <- logisic_regression_prediction(logistic_best_cutoff, logistic, input_data_set)
    best_logistic_accuracy_data <- accuracy_data_results(paste0("Test set logistic regression result at p = ", logistic_best_cutoff), 
    best_logistic_result)
    
    best_logistic_accuracy_data %>% knitr::kable()
#
#
##############################################################################################################

    
############################################################################################################
#
# same predictions made by pca and logistic regression methods

  same_prediction <- best_logistic_result %>%
  inner_join(best_pca_result, by = c("Row_ID", "Amount_Category", "predicted")) %>%
  select(Row_ID, Amount_Category, predicted)
  
  same_prediction_correct <- same_prediction %>% filter(predicted == Amount_Category) 
  
  same_prediction_incorrect <- same_prediction %>% filter(predicted != Amount_Category)

  ################################################################################################

    same_prediction_correct_claim <- same_prediction_correct %>% filter(predicted == "CLAIM")
    
    same_prediction_correct_no_claim <- same_prediction_correct %>% filter(predicted == "NO_CLAIM")
    
    same_prediction_incorrect_claim <- same_prediction_incorrect %>% filter(predicted == "CLAIM")
    
    same_prediction_incorrect_no_claim <- same_prediction_incorrect %>% filter(predicted == "NO_CLAIM") 
    
    #################################################################################################
    
    same_prediction_classifications <- data.frame()
    
    same_prediction_classifications <- rbind(same_prediction_classifications,
    pca_data %>% 
    semi_join(same_prediction_correct_claim, by = "Row_ID") %>%
    mutate(classification = "CORRECT_CLAIM"))
    
    same_prediction_classifications <- rbind(same_prediction_classifications,
    pca_data %>% 
    semi_join(same_prediction_correct_no_claim, by = "Row_ID") %>%
    mutate(classification = "CORRECT_NO_CLAIM"))
    
    same_prediction_classifications <- rbind(same_prediction_classifications,
    pca_data %>% 
    semi_join(same_prediction_incorrect_claim, by = "Row_ID") %>%
    mutate(classification = "INCORRECT_CLAIM"))
    
    same_prediction_classifications <- rbind(same_prediction_classifications,
    pca_data %>% 
    semi_join(same_prediction_incorrect_no_claim, by = "Row_ID") %>%
    mutate(classification = "INCORRECT_NO_CLAIM"))
    
    
    
    same_prediction_classifications %>% ggplot(aes(x = PC1, y = PC2, label = classification)) +
    geom_point(aes(col = classification)) +
    theme_bw() +
    scale_color_discrete(name = "Claim Class") +
    ggtitle("Same Prediction for both PCA and Logistic Analysis")
    
    ################################################################################################# 
    
    different_prediction <- pca_data %>% anti_join(same_prediction, by = "Row_ID") %>%
    select(Row_ID)
#
#
############################################################################################################


############################################################################################################  
#
# pca predictions

    pca_prediction_correct <- best_pca_result %>% semi_join(different_prediction) %>%
    filter(predicted == Amount_Category) %>%
    select(Row_ID, Amount_Category, predicted)
    
    pca_prediction_incorrect <- best_pca_result %>% semi_join(different_prediction) %>%
    filter(predicted != Amount_Category) %>%
    select(Row_ID, Amount_Category, predicted)    
    
    #################################################################################################      
    
    pca_prediction_correct_claim <- pca_prediction_correct %>% filter(predicted == "CLAIM")
    
    pca_prediction_correct_no_claim <- pca_prediction_correct %>% filter(predicted == "NO_CLAIM")
    
    pca_prediction_incorrect_claim <- pca_prediction_incorrect %>% filter(predicted == "CLAIM")
    
    pca_prediction_incorrect_no_claim <- pca_prediction_incorrect %>% filter(predicted == "NO_CLAIM") 
    
    #################################################################################################
    
    pca_prediction_classifications <- data.frame()
    
    pca_prediction_classifications <- rbind(pca_prediction_classifications,
    pca_data %>% 
    semi_join(pca_prediction_correct_claim, by = "Row_ID") %>%
    mutate(classification = "CORRECT_CLAIM"))
    
    pca_prediction_classifications <- rbind(pca_prediction_classifications,
    pca_data %>% 
    semi_join(pca_prediction_correct_no_claim, by = "Row_ID") %>%
    mutate(classification = "CORRECT_NO_CLAIM"))
    
    pca_prediction_classifications <- rbind(pca_prediction_classifications,
    pca_data %>% 
    semi_join(pca_prediction_incorrect_claim, by = "Row_ID") %>%
    mutate(classification = "INCORRECT_CLAIM"))
    
    pca_prediction_classifications <- rbind(pca_prediction_classifications,
    pca_data %>% 
    semi_join(pca_prediction_incorrect_no_claim, by = "Row_ID") %>%
    mutate(classification = "INCORRECT_NO_CLAIM"))
    
    
    
    pca_prediction_classifications %>% ggplot(aes(x = PC1, y = PC2, label = classification)) +
    geom_point(aes(col = classification)) +
    theme_bw() +
    scale_color_discrete(name = "Claim Class") +
    ggtitle("PCA Prediction Analysis")
#
#
##############################################################################################################  


####################################################################################################################  
#
# logistic regression predictions

    logistic_prediction_correct <- best_logistic_result %>% semi_join(different_prediction) %>%
    filter(predicted == Amount_Category) %>%
    select(Row_ID, Amount_Category, predicted)
    
    logistic_prediction_incorrect <- best_logistic_result %>% semi_join(different_prediction) %>%
    filter(predicted != Amount_Category) %>%
    select(Row_ID, Amount_Category, predicted)    
    
    #################################################################################################      
    
    logistic_prediction_correct_claim <- logistic_prediction_correct %>% filter(predicted == "CLAIM")
    
    logistic_prediction_correct_no_claim <- logistic_prediction_correct %>% filter(predicted == "NO_CLAIM")
    
    logistic_prediction_incorrect_claim <- logistic_prediction_incorrect %>% filter(predicted == "CLAIM")
    
    logistic_prediction_incorrect_no_claim <- logistic_prediction_incorrect %>% filter(predicted == "NO_CLAIM") 
    
    ##################################################################################################  
    
    logistic_prediction_classifications <- data.frame()
    
    logistic_prediction_classifications <- rbind(logistic_prediction_classifications,
    pca_data %>% 
    semi_join(logistic_prediction_correct_claim, by = "Row_ID") %>%
    mutate(classification = "CORRECT_CLAIM"))
    
    logistic_prediction_classifications <- rbind(logistic_prediction_classifications,
    pca_data %>% 
    semi_join(logistic_prediction_correct_no_claim, by = "Row_ID") %>%
    mutate(classification = "CORRECT_NO_CLAIM"))
    
    logistic_prediction_classifications <- rbind(logistic_prediction_classifications,
    pca_data %>% 
    semi_join(logistic_prediction_incorrect_claim, by = "Row_ID") %>%
    mutate(classification = "INCORRECT_CLAIM"))
    
    logistic_prediction_classifications <- rbind(logistic_prediction_classifications,
    pca_data %>% 
    semi_join(logistic_prediction_incorrect_no_claim, by = "Row_ID") %>%
    mutate(classification = "INCORRECT_NO_CLAIM"))
    
    
    
    logistic_prediction_classifications %>% ggplot(aes(x = PC1, y = PC2, label = classification)) +
    geom_point(aes(col = classification)) +
    theme_bw() +
    scale_color_discrete(name = "Claim Class") +
    ggtitle("Logistic Prediction Analysis")
#
#
###############################################################################################################



############################################################################################################
#
# correct claim predictions

    correct_claim_predictions <- data.frame()
    
    correct_claim_predictions <- rbind(correct_claim_predictions,
    same_prediction_classifications %>% 
    filter(classification == "CORRECT_CLAIM") %>%
    mutate(algorithm = "PCA_AND_LOGISTIC"))
    
    correct_claim_predictions <- rbind(correct_claim_predictions,
    pca_prediction_classifications %>% 
    filter(classification == "CORRECT_CLAIM") %>%
    mutate(algorithm = "PCA")) 
    
    correct_claim_predictions <- rbind(correct_claim_predictions,
    logistic_prediction_classifications %>% 
    filter(classification == "CORRECT_CLAIM") %>%
    mutate(algorithm = "LOGISTIC"))  
    
    correct_claim_predictions %>% 
    ggplot(aes(x = PC1, y = PC2, label = algorithm)) +
    geom_jitter(aes(col = algorithm)) +
    theme_bw() +
    scale_color_discrete(name = "Algorithm") +
    ggtitle("Actual Claims Predicted by Algorithm") 
#
#
###########################################################################################################

    
###########################################################################################################
#
# correct no claim predictions

    correct_no_claim_predictions <- data.frame()
    
    correct_no_claim_predictions <- rbind(correct_no_claim_predictions,
    same_prediction_classifications %>% 
    filter(classification == "CORRECT_NO_CLAIM") %>%
    mutate(algorithm = "PCA_AND_LOGISTIC"))
    
    correct_no_claim_predictions <- rbind(correct_no_claim_predictions,
    pca_prediction_classifications %>% 
    filter(classification == "CORRECT_NO_CLAIM") %>%
    mutate(algorithm = "PCA")) 
    
    correct_no_claim_predictions <- rbind(correct_no_claim_predictions,
    logistic_prediction_classifications %>% 
    filter(classification == "CORRECT_NO_CLAIM") %>%
    mutate(algorithm = "LOGISTIC"))  
    
    correct_no_claim_predictions %>% 
    ggplot(aes(x = PC1, y = PC2, label = algorithm)) +
    geom_point(aes(col = algorithm)) +
    theme_bw() +
    scale_color_discrete(name = "Algorithm") +
    ggtitle("Actual No Claims Predicted by Algorithm") 
#
#
###########################################################################################################


###########################################################################################################
#
# incorrect no claim predictions

    incorrect_no_claim_predictions <- data.frame()
    
    incorrect_no_claim_predictions <- rbind(incorrect_no_claim_predictions,
    same_prediction_classifications %>% 
    filter(classification == "INCORRECT_NO_CLAIM") %>%
    mutate(algorithm = "PCA_AND_LOGISTIC"))
    
    incorrect_no_claim_predictions <- rbind(incorrect_no_claim_predictions,
    pca_prediction_classifications %>% 
    filter(classification == "INCORRECT_NO_CLAIM") %>%
    mutate(algorithm = "PCA")) 
    
    incorrect_no_claim_predictions <- rbind(incorrect_no_claim_predictions,
    logistic_prediction_classifications %>% 
    filter(classification == "INCORRECT_NO_CLAIM") %>%
    mutate(algorithm = "LOGISTIC"))  
    
    incorrect_no_claim_predictions %>% 
    ggplot(aes(x = PC1, y = PC2, label = algorithm)) +
    geom_point(aes(col = algorithm)) +
    theme_bw() +
    scale_color_discrete(name = "Algorithm") +
    ggtitle("Incorrect No Claims Predicted by Algorithm") 
#
#
############################################################################################################


###########################################################################################################
#
# Incorrect predictions

    incorrect_claim_predictions <- data.frame()
    
    incorrect_claim_predictions <- rbind(incorrect_claim_predictions,
    same_prediction_classifications %>% 
    filter(classification == "INCORRECT_CLAIM") %>%
    mutate(algorithm = "PCA_AND_LOGISTIC"))
    
    incorrect_claim_predictions <- rbind(incorrect_claim_predictions,
    pca_prediction_classifications %>% 
    filter(classification == "INCORRECT_CLAIM") %>%
    mutate(algorithm = "PCA")) 
    
    incorrect_claim_predictions <- rbind(incorrect_claim_predictions,
    logistic_prediction_classifications %>% 
    filter(classification == "INCORRECT_CLAIM") %>%
    mutate(algorithm = "LOGISTIC"))  
    
    incorrect_claim_predictions %>% 
    ggplot(aes(x = PC1, y = PC2, label = algorithm)) +
    geom_jitter(aes(col = algorithm)) +
    theme_bw() +
    scale_color_discrete(name = "Algorithm") +
    ggtitle("Incorrect Claims Predicted by Algorithm")
#
#
##################################################################################################################


##########################################################################################
#  
#  Households with No Claims

# all 2005 households
    claim_data_2005_households <- claim_data_2005 %>% 
    group_by(Household_ID) %>%
    summarize(n = n())
    
    # households with claims in 2005
    claim_data_2005_households_with_claims <- claim_data_2005 %>% 
    filter(Amount_Category == "CLAIM") %>%
    group_by(Household_ID) %>%
    summarize(n = n())  
    
    # households with no claims in 2005
    claim_data_2005_households_with_no_claims <- claim_data_2005_households %>% 
    anti_join(claim_data_2005_households_with_claims, by = "Household_ID") %>%
    group_by(Household_ID) %>%
    summarize(n = n())  
    
    # households with claims in 2006 where no claims existed for those households in 2005
    claim_data_2006_households_with_claims_which_had_no_claims_in_2005 <- claim_data_2006 %>% 
    inner_join(claim_data_2005_households_with_no_claims) %>%
    filter(Amount_Category == "CLAIM") %>%
    group_by(Household_ID) %>%
    summarize(n = n())
    
    # households with no claims in 2006 where no claims existed for those households in 2005
    claim_data_2006_households_with_no_claims_which_had_no_claims_in_2005 <- claim_data_2006 %>% 
    inner_join(claim_data_2005_households_with_no_claims) %>%
    filter(Amount_Category == "NO_CLAIM") %>%
    group_by(Household_ID) %>%
    summarize(n = n())
    
    # counts of claims and no claim vehicles for households with no claims in 2005 
    chi_square_data_2005_2006_for_2005_no_claims <- 
    matrix(c(0,
    sum(claim_data_2005_households_with_no_claims$n),
    sum(claim_data_2006_households_with_claims_which_had_no_claims_in_2005$n),
    sum(claim_data_2006_households_with_no_claims_which_had_no_claims_in_2005$n)), nrow = 2)
    
    colnames(chi_square_data_2005_2006_for_2005_no_claims) <- c("2005", "2006")
    rownames(chi_square_data_2005_2006_for_2005_no_claims) <- c("Claim", "No Claim")
    
    chi_square_data_2005_2006_for_2005_no_claims %>% knitr::kable()
    chisq.test(chi_square_data_2005_2006_for_2005_no_claims)
#
#
##############################################################################################################


##############################################################################################################
#
#  Households with Claims in 2005

    # 2006 claim data for 2005 households with claims
    claim_data_2006_for_households_with_claims_in_2005 <- claim_data_2006 %>% 
    left_join(claim_data_2005_households_with_claims) %>%
    group_by(Household_ID) %>%
    summarize(n = n())
    
    # households with claims in 2006
    claim_data_2006_households_with_claims <- claim_data_2006 %>% 
    filter(Amount_Category == "CLAIM") %>%
    group_by(Household_ID) %>%
    summarize(n = n())
    
    # households with claims in 2005 with no claim vehicles in 2005
    claim_data_2005_households_with_2005_claims_with_2005_no_claim_vehicles <- claim_data_2005 %>% 
    inner_join(claim_data_2005_households_with_claims) %>%
    filter(Amount_Category == "NO_CLAIM") %>%  
    group_by(Household_ID) %>%
    summarize(n = n())
    
    # households with claims in 2005 with no claim vehicles in 2006
    claim_data_2006_households_with_2005_claims_with_2006_no_claim_vehicles <- claim_data_2006 %>% 
    inner_join(claim_data_2005_households_with_claims) %>%
    filter(Amount_Category == "NO_CLAIM") %>%  
    group_by(Household_ID) %>%
    summarize(n = n())  
    
    # counts of claims and no claim vehicles for households with claims in 2005 
    chi_square_data_2005_2006_for_2005_claims <- 
    matrix(c(sum(claim_data_2005_households_with_claims$n),
    sum(claim_data_2005_households_with_2005_claims_with_2005_no_claim_vehicles$n),
    sum(claim_data_2006_households_with_claims$n),
    sum(claim_data_2006_households_with_2005_claims_with_2006_no_claim_vehicles$n)), nrow = 2)
    
    # counts of claims and no claim vehicles for households with claims in 2005 
    chi_square_data_2005_2006_for_2005_claims <- matrix(c(sum(claim_data_2005_households_with_claims$n),
    sum(claim_data_2005_households_with_2005_claims_with_2005_no_claim_vehicles$n),
    sum(claim_data_2006_households_with_claims$n),
    sum(claim_data_2006_households_with_2005_claims_with_2006_no_claim_vehicles$n)),nrow = 2)
    
    colnames(chi_square_data_2005_2006_for_2005_claims) <- c("2005", "2006")
    rownames(chi_square_data_2005_2006_for_2005_claims) <- c("Claim", "No Claim")
    
    chi_square_data_2005_2006_for_2005_claims %>% knitr::kable()
    chisq.test(chi_square_data_2005_2006_for_2005_claims)
#
#
#################################################################################################################


#################################################################################################################
#
#  Claims in 2006 for households with no claims in 2005

    household_claim_predictions_2005_no_claim_2006_claim <- data.frame()
    
    household_claim_predictions_2005_no_claim_2006_claim <- rbind(household_claim_predictions_2005_no_claim_2006_claim, 
    claim_data_2006 %>% 
    semi_join(claim_data_2006_households_with_claims_which_had_no_claims_in_2005) %>%
    select(Row_ID) %>%
    inner_join(same_prediction_classifications, on = "Row_ID") %>% 
    filter(classification == "CORRECT_CLAIM") %>%
    mutate(algorithm = "PCA_AND_LOGISTIC"))
    
    household_claim_predictions_2005_no_claim_2006_claim <- rbind(household_claim_predictions_2005_no_claim_2006_claim, 
    claim_data_2006 %>% 
    semi_join(claim_data_2006_households_with_claims_which_had_no_claims_in_2005) %>%
    select(Row_ID) %>%
    inner_join(pca_prediction_classifications, on = "Row_ID") %>% 
    filter(classification == "CORRECT_CLAIM") %>%
    mutate(algorithm = "PCA")) 
    
    household_claim_predictions_2005_no_claim_2006_claim <- rbind(household_claim_predictions_2005_no_claim_2006_claim, 
    claim_data_2006 %>% 
    semi_join(claim_data_2006_households_with_claims_which_had_no_claims_in_2005) %>%
    select(Row_ID) %>%
    inner_join(logistic_prediction_classifications, on = "Row_ID") %>% 
    filter(classification == "CORRECT_CLAIM") %>%
    mutate(algorithm = "LOGISTIC"))   
    
    household_claim_predictions_2005_no_claim_2006_claim <- rbind(household_claim_predictions_2005_no_claim_2006_claim, 
    claim_data_2006 %>% 
    inner_join(claim_data_2006_households_with_claims_which_had_no_claims_in_2005) %>%
    select(Row_ID) %>%
    inner_join(pca_data) %>%  
    filter(Category == "CLAIM") %>%  
    anti_join(correct_claim_predictions, on = "Row_ID") %>%
    mutate(classification = "INCORRECT_NO_CLAIM", algorithm = "ACTUAL_NOT_PREDICTED")) 
    
    household_claim_predictions_2005_no_claim_2006_claim %>% 
      ggplot(aes(x = PC1, y = PC2, label = algorithm)) +
      geom_point(aes(col = algorithm)) +
      theme_bw() +
      scale_color_discrete(name = "Algorithm") +
      ggtitle("Claims in 2006 for Households with No Claims in 2005")
#
#
###################################################################################################


##############################################################################################
#
#  No Claims in 2006 for households with no claims in 2005

    household_claim_predictions_2005_no_claim_2006_no_claim <- data.frame()
    
    household_claim_predictions_2005_no_claim_2006_no_claim <- rbind(household_claim_predictions_2005_no_claim_2006_no_claim, 
    claim_data_2006 %>% 
    semi_join(claim_data_2006_households_with_no_claims_which_had_no_claims_in_2005) %>%
    select(Row_ID) %>%
    inner_join(same_prediction_classifications, on = "Row_ID") %>% 
    filter(classification == "CORRECT_NO_CLAIM") %>%
    mutate(algorithm = "PCA_AND_LOGISTIC"))
    
    household_claim_predictions_2005_no_claim_2006_no_claim <- rbind(household_claim_predictions_2005_no_claim_2006_no_claim, 
    claim_data_2006 %>% 
    semi_join(claim_data_2006_households_with_no_claims_which_had_no_claims_in_2005) %>%
    select(Row_ID) %>%
    inner_join(pca_prediction_classifications, on = "Row_ID") %>% 
    filter(classification == "CORRECT_NO_CLAIM") %>%
    mutate(algorithm = "PCA")) 
    
    household_claim_predictions_2005_no_claim_2006_no_claim <- 
    rbind(household_claim_predictions_2005_no_claim_2006_no_claim, 
    claim_data_2006 %>% 
    semi_join(claim_data_2006_households_with_no_claims_which_had_no_claims_in_2005) %>%
    select(Row_ID) %>%
    inner_join(logistic_prediction_classifications, on = "Row_ID") %>% 
    filter(classification == "CORRECT_NO_CLAIM") %>%
    mutate(algorithm = "LOGISTIC"))   
    
    household_claim_predictions_2005_no_claim_2006_no_claim <- 
    rbind(household_claim_predictions_2005_no_claim_2006_no_claim, 
    claim_data_2006 %>% 
    inner_join(claim_data_2006_households_with_no_claims_which_had_no_claims_in_2005) %>%
    select(Row_ID) %>%
    inner_join(pca_data) %>%  
    filter(Category == "NO_CLAIM") %>%  
    anti_join(correct_claim_predictions, on = "Row_ID") %>%
    mutate(classification = "INCORRECT_CLAIM", algorithm = "ACTUAL_NOT_PREDICTED")) 
    
    
    household_claim_predictions_2005_no_claim_2006_no_claim %>% 
      ggplot(aes(x = PC1, y = PC2, label = algorithm)) +
      geom_point(aes(col = algorithm)) +
      theme_bw() +
      scale_color_discrete(name = "Algorithm") +
      ggtitle("No Claims in 2006 for Households with No Claims in 2005")
#
#
######################################################################################################

 
#################################################################################################
#
#   Predicting No Claim in 2006 for households not having claims in 2005

    best_pca_result_with_past_year_no_claim_correction <- best_pca_result %>% 
    semi_join(household_claim_predictions_2005_no_claim_2006_no_claim, on = "Row_ID") %>%
    filter(predicted == "CLAIM") %>%
    mutate(predicted = "NO_CLAIM")
    best_pca_result_claim <- anti_join(best_pca_result, best_pca_result_with_past_year_no_claim_correction, by = "Row_ID")
    best_pca_result_with_past_year_no_claim_correction <- rbind(best_pca_result_with_past_year_no_claim_correction,
    best_pca_result_claim)
    best_pca_result_with_past_year_no_claim_correction$predicted <- 
    factor(best_pca_result_with_past_year_no_claim_correction$predicted, levels = c("CLAIM", "NO_CLAIM")) 
    pca_with_past_year_no_claims_accuracy_data <- accuracy_data_results("PCA best prediction with past year no claim correction", 
      best_pca_result_with_past_year_no_claim_correction)
    pca_with_past_year_no_claims_accuracy_data %>% knitr::kable()
#
#
######################################################################################################


#################################################################################################
#
#   Adding Predictions of Claim by logistic regression method to the PCA best results in region 
#   PCA approach does not predict as Claim

    logistic_boost <- 
    pca_data %>% 
    semi_join(best_logistic_result, on = "Row_ID") %>%
    filter(PC1 <= pca_best_pc1 | PC2 <= pca_best_pc2) %>%
    select(Row_ID) %>%
    inner_join(best_logistic_result) 
    predictions_from_pca <-  
    pca_data %>% 
    semi_join(best_pca_result, on = "Row_ID") %>%
    filter(PC1 > pca_best_pc1 & PC2 > pca_best_pc2) %>%
    select(Row_ID) %>%
    inner_join(best_pca_result) %>%
    mutate(predicted = "CLAIM")
    best_pca_result_with_logistic_boost <- rbind(predictions_from_pca, logistic_boost)
    best_pca_result_with_logistic_boost$predicted <- 
    factor(best_pca_result_with_logistic_boost$predicted, levels = c("CLAIM", "NO_CLAIM")) 
    best_pca_result_with_logistic_boost$Amount_Category <- 
    factor(best_pca_result_with_logistic_boost$Amount_Category, levels = c("CLAIM", "NO_CLAIM"))
    pca_with_logistic_boost_accuracy_data <- accuracy_data_results("PCA best prediction with logistic_boost", 
      best_pca_result_with_logistic_boost) 
    pca_with_logistic_boost_accuracy_data %>% knitr::kable()

#    
#
######################################################################################################


###############################################################################################
#
#   Predicting with logistic boost and No Claim in 2006 for households not having claims in 2005

    best_pca_result_with_logistic_boost_and_past_year_no_claim_correction <- best_pca_result_with_logistic_boost %>% 
    semi_join(household_claim_predictions_2005_no_claim_2006_no_claim, on = "Row_ID") %>%
    filter(predicted == "CLAIM") %>%
    mutate(predicted = "NO_CLAIM")
    best_pca_result_with_logistic_boost_claim <- anti_join(best_pca_result_with_logistic_boost, 
    best_pca_result_with_logistic_boost_and_past_year_no_claim_correction, by = "Row_ID")
    best_pca_result_with_logistic_boost_and_past_year_no_claim_correction <-   
    rbind(best_pca_result_with_logistic_boost_and_past_year_no_claim_correction, best_pca_result_with_logistic_boost_claim)
    best_pca_result_with_logistic_boost_and_past_year_no_claim_correction$predicted <- 
    factor(best_pca_result_with_logistic_boost_and_past_year_no_claim_correction$predicted, levels = c("CLAIM", "NO_CLAIM")) 
    pca_with_logistic_boost_and_past_year_no_claims_accuracy_data <- 
    accuracy_data_results("PCA, logistic boost with past year no claim prediction", 
                          best_pca_result_with_logistic_boost_and_past_year_no_claim_correction)
    pca_with_logistic_boost_and_past_year_no_claims_accuracy_data %>% knitr::kable()
#
#
###############################################################################################





