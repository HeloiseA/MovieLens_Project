#############################################################
# HarvardX: PH125.9 - Data Science: Capstone
#############################################################
#
# The following script uses a subsample of the MovieLens datasets and tests
# different models in order to predict movie ratings from users.
# This code was run on Windows 8 OS with RStudio Version 1.1.447.
#
# Find this project online at: https://github.com/HeloiseA/MovieLens_Project
#
# The following section was given by the course instructors.
#
#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(dplyr)
library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#############################################################
# Create RMSE function
#############################################################

# Where y_hat is the vector of the predicted ratings

RMSE <- function(validation, y_hat){
  sqrt(mean((validation - y_hat)^2))
}

#############################################################
# Create a first model for the predicted ratings
#############################################################

# All predicted ratings in y_hat1 are the average rating of the edx training set.

y_hat1 <- mean(edx$rating)

# Compute RMSE for model 1

rmse1 <- RMSE(validation$rating, y_hat1)
cat("RMSE from Model 1: ", rmse1)

#############################################################
# Create a second model for the predicted ratings
#############################################################

# Here, we add a parameter to account for the average rating of each movie
# Our average rating from model 1 is renamed "mu"

mu <- mean(edx$rating)

# The movie_avgs data frame contains the movieIds and their parameter b_i,
# where b_i is the mean difference of the average rating mu from each movie rating

movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# The predicted ratings in y_hat2 are based on the mean rating and movie-dependant parameters b_i

y_hat2 <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

# Compute RMSE for model 2

rmse2 <- RMSE(validation$rating, y_hat2)
cat("RMSE from Model 2: ", rmse2)

#############################################################
# Create a third model for the predicted ratings
#############################################################

# Another parameter is added to account for the average rating given by a user

# The user_avgs data frame contains the movieIds and their parameter b_u,
# the mean difference of the average rating mu and movie parameter b_i from each user rating

user_avgs <- edx %>% left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# The predicted ratings in y_hat3 are based on the mean rating, movie-dependant parameters b_i,
# and user-dependant parameters b_u

y_hat3 <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Compute RMSE for model 3

rmse3 <- RMSE(validation$rating, y_hat3)
cat("RMSE from Model 3: ", rmse3)

#############################################################
# Use Regularization principles to improve on the third model
#############################################################

# Here, we use regularization to take into account the number of ratings per movie
# to diminish the b_i effect of movies with a small number of ratings

# Create a grid for the tuning parameter lambda

lambdas <- seq(0, 10, 0.25)

# Create a data frame containing the movieId, its sum of all b_i,
# and its number of ratings

sums_movie <- edx %>%
  group_by(movieId) %>%
  summarize(s = sum(rating - mu), n_i = n())

# Use sapply to test all values of lambdas for the regularization of the movie effect parameter.
# Compute all RMSEs for model 3 with a regularized parameter b_i

rmses <- sapply(lambdas, function(l){
  y_hat4 <- validation %>%
    left_join(sums_movie, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(validation$rating, y_hat4))
})

# Plotting a graph to show the variations of the RMSEs with lambda

library(ggplot2)
library(ggthemes)
library(grid)

p <- qplot(lambdas, rmses, main="Values of RMSE vs. parameter Lambda",
           xlab="Lambda", ylab="RMSE")
p + theme_economist() +
  theme(panel.grid.major = element_line(color = "white", size = 0.5),
        panel.grid.minor = element_line(color = "white", size = 0.5))

# Display minimum RMSE value and associated lambda parameter

final_rmse <- min(rmses)
cat("Minimum RMSE from model 3 with regularized movie effect parameter: ", final_rmse)
cat("RMSE obtained with lambda = ", lambdas[which.min(rmses)])
