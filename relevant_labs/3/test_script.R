library(caret)

# Load data as in the Rmd file
interestsEEUU <- read.table("./Data/MEI_FIN_23092019161922426.csv", sep=",", header=TRUE)
interestsEEUU <- ts(interestsEEUU[interestsEEUU$Country=="United States" & interestsEEUU$Tipo=="IRLT", c("Value")], start=2007, frequency=12)

# 2. Sliding window function
create_sliding_window <- function(ts_data, window_size, step_ahead = 1) {
  n <- length(ts_data)
  num_rows <- n - window_size - step_ahead + 1
  X <- matrix(NA, nrow = num_rows, ncol = window_size)
  Y <- numeric(num_rows)
  
  for (i in 1:num_rows) {
    X[i, ] <- ts_data[i:(i + window_size - 1)]
    Y[i] <- ts_data[i + window_size + step_ahead - 1]
  }
  
  df <- data.frame(X, Y = Y)
  colnames(df)[1:window_size] <- paste0("Lag_", window_size:1)
  return(df)
}

# Apply function
w <- 12 # window size
df <- create_sliding_window(interestsEEUU, w)

# 1. Split data (70% train, 30% test) keeping temporal order
n_train <- floor(0.7 * nrow(df))
train_data <- df[1:n_train, ]
test_data <- df[(n_train + 1):nrow(df), ]

# Apply regression algorithm from caret
# We can use timeslices for caret CV
time_control <- trainControl(method = "timeslice",
                              initialWindow = 48,
                              horizon = 12,
                              fixedWindow = FALSE)

set.seed(42)
lm_model <- train(Y ~ ., data = train_data, method = "lm", trControl = time_control)
knn_model <- train(Y ~ ., data = train_data, method = "knn", trControl = time_control)

# Evaluate RMSE on test set
lm_pred <- predict(lm_model, newdata = test_data)
knn_pred <- predict(knn_model, newdata = test_data)

cat("LM RMSE:", RMSE(lm_pred, test_data$Y), "\n")
cat("KNN RMSE:", RMSE(knn_pred, test_data$Y), "\n")

